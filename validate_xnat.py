import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.MIMEBase import MIMEBase
from email import Encoders
import os
import os.path as osp
import hashlib
import string
import pandas as pd
from xml import etree
from datetime import datetime
from pyxnat import Interface
from xml import etree
import pandas as pd

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

class Mail(object):
    def __init__(self, page_html = ''):
        self.message = MIMEMultipart()
        self.message.attach(MIMEText(page_html.encode('utf-8'), 'html','utf-8'))
        self.email_sender = 'xnat@fpmaragall.org'

    def send_mail(self, receiver_mail, subject='XNAT update', attached_files=[]):
        '''Function to send a mail'''
        for f in attached_files:
             part = MIMEBase('application', "octet-stream")
             part.set_payload( open(f, "rb").read() )
             Encoders.encode_base64(part)
             part.add_header('Content-Disposition', 'attachment', filename="%s" % os.path.basename(f))
             self.message.attach(part)

        self.message['Subject'] = subject
        self.message['From'] = self.email_sender
        self.message['To'] =  ", ".join(receiver_mail)
        s = smtplib.SMTP(host='smtp.gmail.com', port=587)
        s.ehlo()
        s.starttls()
        s.login('xnat@fpmaragall.org', 'Xnat2016')

        s.sendmail(self.email_sender, receiver_mail, self.message.as_string())
        s.quit()


def validate_session(fp, project, xnatId, verbose=False):

    log = []
    df = pd.read_excel(fp)
    return validate_session_from_df(df, project, xnatId, verbose)

def resolve_aliases(l, aliases, log=[]):
    new_list = []
    transtab = {}
    for e in l:
        has_alias = False
        for k, v in aliases.items():
            if e in v:
                new_list.append(k)
                log.append('%s is found as %s'%(k, e))
                transtab[k] = e
                has_alias = True
        if not has_alias:
            new_list.append(e)
            transtab[e] = e
    return new_list, transtab


def validate_session_from_df(df, project, xnatId, verbose=False):
    result = 0 # 0: OK   1: Warning   2:Error
    log = []

    # Checks emptiness before starting anything
    df2 = df[df['xnatId'] == xnatId][['ID', 'type', 'frames', 'file_count']]
    session_seqlist = df2['type'].tolist() # List of sequences of the session

    if df2.empty:
        log.append('%s not found'%xnatId)
        result = 2
        return result

    rules = {}
    rules['ALFA_OPCIONAL'] = {
        'obligatory': {'T1_ALFA1':240,
                       'rRS_ALFA1 SENSE':230,
                       'RS_ALFA1 SENSE':13800,
                       'FLAIR_ALFA1':180,
                       'T2_ALFA1':360,
                       'IR_ALFA1':78,
                       'rDWI_ALFA1':66,
                       'DWI_ALFA1':4752,
                       'SE-fMRI-AP SENSE':225,
                       'SE-fMRI-PA SENSE':225},
        'aliases' : { 'rRS_ALFA1 SENSE':['rRS_ALFA1'],
                      'SE-fMRI-AP SENSE':['SE-fMRI-AP'],
                      'RS_ALFA1 SENSE':['RS_ALFA1'],
                      'SE-fMRI-PA SENSE':['SE-fMRI-PA'],
                      'IR_ALFA1':['IR_ALFA1 SENSE'],
                      'T1_CORONAL':['CORONAL', 'T1 CORONAL', 'T1_Coronal'],
                      'T1_AXIAL':['AXIAL', 'T1 AXIAL', 'T1_Axial'],
                      'FLAIR_CORONAL':['FLAIR CORONAL', 'Coronal FLAIR'],
                      'FLAIR_AXIAL':['FLAIR AXIAL', 'Axial FLAIR']},
        'warning_if_missing' : ['T1_CORONAL', 'T1_AXIAL', 'FLAIR_CORONAL', 'FLAIR_AXIAL'],
        'silently_identified' : ['B1_calibration', 'B0_ALFA1', 'SmartBrain SENSE',
            'Patient Aligned MPR AWPLAN_SMARTPLAN_TYPE_BRAIN'] }

    rules['EPAD'] = {
        'obligatory': {'MPRAGE_SENSE2': 170,
                       'FLAIR_SAG': 192,
                       'Axial T2-Star':47,
                       'Axial T2W-TSE with Fat Sat':47,
                       'T1_ALFA1':240},
        'aliases':{},
        'warning_if_missing': ['MPRAGE_CORONAL', 'MPRAGE_SAGITAL', 'FLAIR_CORONAL',
            'FLAIR_AXIAL'],
        'silently_identified': ['B1_calibration', 'Survey']}

    rules['ALFA_PLUS2'] =  {
        'obligatory' : {'sT1W_3D_TFE_HR_32 iso1.2 long AT':150,
                        'M0_ASL':30,
                        'pCASL':1800,
                        'sWIP pCASL SENSE':30,
                        'SV_PRESS_100_Myo_CHESS_Hipo':2,
                        'SV_PRESS_100_Myo_CHESS_Ang':2,
                        'SV_PRESS_100_Myo_CHESS_Cun':2,
                        'QFLOW_CSF':90,
                        'QFLOW_CAROTID':90,
                        'WIP muti-b bajo max':64,
                        'WIP muti-b medio max':128,
                        'WIP muti-b altos max':96,
                        'SWIp SENSE': 280,
                        'DelRec - SWIp SENSE':1120},
        'aliases':{'WIP muti-b bajo max':['WIP muti-b  bajo max']},

        'warning_if_missing' : ['VWIP sT1W_3D_TFE_HR_32 iso1.2 long AT SENSE',
            'VWIP sT1W_3D_axial', 'VWIP sT1W_3D_coronal'],
        'silently_identified' : ['SmartBrain SENSE']}

    rules['ALFA_PLUS'] = rules['ALFA_OPCIONAL']

    # Dropping mock sequences
    for index, row in df2.iterrows():
        if row['ID'].startswith('0-') or 'OT' in row['ID']:
            log.append('Ignoring sequence %s'%row['ID'])
            session_seqlist.remove(row['type'])
            df2.drop(index, inplace=True)

    #Resolving aliases
    session_seqlist, transtab = resolve_aliases(session_seqlist, rules[project]['aliases'], log)

    print session_seqlist, transtab
    # Checking obligatory
    for k, v in rules[project]['obligatory'].items():
        key = 'frames' if not 'CHESS' in transtab[k] else 'file_count'
        if session_seqlist.count(k) == 1 and df2[df2['type'] == transtab[k]][key].iloc[0] != str(v):
            log.append('*** %s is unique but has wrong file_count (%s instead of %s) ***'%(k, df2[df2['type'] == transtab[k]]['frames'].iloc[0], v))
            result = 2
        if session_seqlist.count(k) > 1:
            log.append('%s multiple occurrences (%s)'%(k, df2[df2['type']==transtab[k]]['ID'].tolist()))
            result = max(result, 1)

    log.append('<br> ===== Missing ========')
    for k, v in rules[project]['obligatory'].items():
        if not k in session_seqlist:
            log.append('*** %s is obligatory and missing ***'%k)
            result = max(result, 2)
    for k in rules[project]['warning_if_missing']:
        if not k in session_seqlist:
            log.append('%s missing'%k)
            result = max(result, 1)

    log.append('<br> ========= Not identified ==========')

    seq = rules[project]['obligatory'].keys()
    seq.extend(rules[project]['warning_if_missing'])
    seq.extend(rules[project]['silently_identified'])
    for k,v in rules[project]['aliases'].items():
        seq.extend(v)
    for each in set(session_seqlist).difference(seq):
        log.append('%s not identified'%each)

    log.append('<br> ========= Silently identified ==========')
    for each in session_seqlist:
        if each in rules[project]['silently_identified'] or each.startswith('0-')\
                or 'OT' in each:
            log.append('%s silently identified'%each)

    return result, log


def collect_sequences_list(central, project, xnatId):
    xml = central._exec(central.select.project(project).experiment(str(xnatId))._uri +
    '?format=xml');
    root = etree.ElementTree.XML(xml)
    data = []
    NS="{http://nrg.wustl.edu/xnat}"
    for s in root.findall(NS+"scans"):
        for scan in s.findall(NS+"scan"):
            d = {}
            d['ID'] = scan.attrib["ID"]
            f = scan.find(NS+'series_description')
            d['series_description'] = f.text if not f is None else 'not found'

            d['type'] = scan.attrib["type"]
            d['file_count'] = 'not found'
            f = scan.find(NS+'frames')
            d['frames'] = f.text if not f is None else 'not found'
            for f in scan.findall(NS+"file"):
                if f.attrib['label'] == 'DICOM':
                    for each in ['label', 'URI', 'format', 'file_count', 'file_size']:
                       d[each] = f.attrib[each]
                elif f.attrib['label'] == 'secondary' and 'CHESS' in d['type']:
                    d['file_count'] = f.attrib['file_count']

            data.append(d)
    return data


def build_mail(project, xnatId):
    startTime = datetime.now()

    config_file = '/home/grg/.xnat_bsc.cfg'
    central = Interface(config=config_file, verify=False)
    projects = central.select.projects().get()
    exp = central.select.project(project).experiment(xnatId)

    # Fetch subject
    xml = central._exec(central.select.project(project).experiment(str(xnatId))._uri +
    '?format=xml');
    NS="{http://nrg.wustl.edu/xnat}"
    root = etree.ElementTree.XML(xml)
    subject = root.find(NS+'dcmPatientId').text

    # Collects session information
    keys = ['ID', 'type', 'file_count', 'frames']
    d = collect_sequences_list(central, project, xnatId)
    data = []
    for each in d:
        row = [project, xnatId]
        for k in keys:
            row.append(each[k])
        data.append(row)
    df = pd.DataFrame(data, columns=['project', 'xnatId', 'ID', 'type', 'file_count', 'frames'])

    # Validates session
    res, log = validate_session_from_df(df, project, xnatId, True)

    # Compiles email body
    html = '''<p><img src="https://static.wixstatic.com/media/eaaa95_b9387dba593a4afdb6ba3e1a1fc3a244.png/v1/fill/w_235,h_52,al_c,usm_0.66_1.00_0.01/eaaa95_b9387dba593a4afdb6ba3e1a1fc3a244.png" alt="" width="235" height="52" /></p>
        <p>&nbsp;</p>
        <p style="color: #222222; font-family: arial, sans-serif; font-size: 12.8px;">Dear XNAT user,</p>
        <p style="color: #222222; font-family: arial, sans-serif; font-size: 12.8px;"><b>THIS IS A TEST.</b></p>
        <p style="color: #222222; font-family: arial, sans-serif; font-size: 12.8px;">The following session was archived in BBRC:</p>
        <ul style="color: #222222; font-family: arial, sans-serif; font-size: 12.8px;">
        <li style="margin-left: 15px;">Project: !!PROJECT!!</li>
        <li style="margin-left: 15px;">Subject: !!SUBJECT!!</li>
        <li style="margin-left: 15px;">Session: !!SESSION!!</li>
        </ul>
        <div><span style="color: #222222; font-family: arial, sans-serif; font-size: 12.8px;">The result of the validation is: </span><span style="color:!!COLOR!!">!!RESULT!!</span></div>
        <div><span style="color: #222222; font-family: arial, sans-serif; font-size: 12.8px;"><br /></span></div>
        <div><span style="color: #222222; font-family: arial, sans-serif; font-size: 12.8px;">The following observations were returned:</span></div>
        <div><span style="color: #222222; font-family: arial, sans-serif; font-size: 12.8px;"><br /></span></div>
        <div style="padding-left: 30px;"><span style="color: #909090; font-family: arial, sans-serif; font-size: 12.8px;">!!REPORT!!</span></div>
        <div><span style="color: #222222; font-family: arial, sans-serif; font-size: 12.8px;"><br /></span></div>
        <div><span style="color: #222222; font-family: arial, sans-serif;">Additional information <a href="!!URL!!">here</a>.<br></span></div>

        <div><span style="color: #222222; font-family: arial, sans-serif; font-size: 12.8px;">BBRC Team</span></div>
        <div><em><span style="color: #909090; font-family: arial, sans-serif;"><span style="font-size: 12.8px;">--<br> generated in !!ELAPSEDTIME!! (!!FILE!! v.!!VERSION!!)</span></span></em></div>
        <p>&nbsp;</p>
        '''

    html = html.replace('!!PROJECT!!', project)
    html = html.replace('!!SUBJECT!!', subject)
    html = html.replace('!!SESSION!!', str(xnatId))
    html = html.replace('!!RESULT!!', ['SUCCESS', 'WARNING', 'ERRORS'][res])
    html = html.replace('!!COLOR!!', ['#229c06', '#e1be2e', '#a11111'][res])
    html = html.replace('!!REPORT!!', '<br>'.join(log))
    e = central.select.experiment(xnatId)
    url = 'http://bscmaragall01.bsc.es:8080/data/experiments/%s?format=html'%e.id()
    html = html.replace('!!URL!!', url)

    # Estimates elapsed time
    seconds = datetime.now() - startTime
    m, s = divmod(seconds.total_seconds(), 60)
    h, m = divmod(m, 60)
    elapsedtime = "%d:%02d:%02d" % (h, m, s)
    version = md5(__file__)[:8]
    html = html.replace('!!ELAPSEDTIME!!', elapsedtime)
    html = html.replace('!!FILE!!', osp.basename(__file__))
    html = html.replace('!!VERSION!!', version)

    # Sends email
    m = Mail(html)
    subject = 'XNAT update: %s (%s) validation results'%(xnatId, project)
    m.send_mail(['goperto@fpmaragall.org'], subject=subject)
