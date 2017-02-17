from pyxnat import Interface
config_file = '/home/grg/.xnat.cfg'
central = Interface(config=config_file, verify=False)
projects = central.select.projects().get()

project = 'EPAD'

# Collecting data from XNAT
subjects = central.select.project(project).subjects()
info = []
for s in subjects:
    try:
        sid = s.get().split('xnat:dcmPatientId>')[1].split('</')[0]
        sname = s.get().split('xnat:dcmPatientName>')[1].split('</')[0]
        accn = s.get().split('xnat:dcmAccessionNumber>')[1].split('</')[0]
        info.append([sid, sname, accn])
    except IndexError:
        print s.label(), s.id(), 'error'

# Building DataFrame
import pandas as pd
df = pd.DataFrame(info, columns=('PatientId', 'PatientName', 'AccessionNumber'))
df = pd.DataFrame(info, index=df['PatientId'], columns=('PatientId', 'PatientName', 'AccessionNumber'))
df = df.sort_index()
del df['PatientId']
df.head()

# Saving it in an Excel table
ew = pd.ExcelWriter('/tmp/%s.xls'%project, encoding='utf-8')
df.to_excel(ew)
ew.save()
