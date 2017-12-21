#!/home/grg/jupyter/bin/python
import os
import sys
import subprocess
import json
import os.path as osp
from glob import glob
import tempfile
import string

from datetime import datetime
startTime = datetime.now()

def size_to_human(full_size):
  size = full_size
  if size >= 1024:
    unit = 'KiB'
    size /= 1024.0
    if size >= 1024:
      unit = 'MiB'
      size /= 1024.0
      if size >= 1024:
        unit = 'GiB'
        size /= 1024.0
        if size >= 1024:
          unit = 'TiB'
          size /= 1024.0
    s = '%.2f' % (size,)
    if s.endswith( '.00' ): s = s[:-3]
    elif s[-1] == '0': s = s[:-1]
    return s + ' ' + unit
  else:
    return str(size)

#===============================================================================
try:

    # Free disk space (df)
    df = subprocess.Popen(["df", "-h", "/media/Temporary"], stdout=subprocess.PIPE)
    output = df.communicate()[0]
    device, size, used, available, percent, mountpoint = \
        output.split("\n")[1].split()

    #====================
    # Runs the full folder scan over /media/Projects and /media/Temporary
    sizes = []
    wds = ['/media/Temporary'] # Folder to scan
    #ddbd = '/home/grg/git/directory_database' # Path to directory_database

    for wd in wds:
        print wd
        cmd = ['du', '-s', wd]
        print cmd
        # Free disk space
        df = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        output = df.communicate()[0]
        print output
        s = size_to_human(string.atof(output.split('\t')[0])*1024.0)
        print wd, s
        sizes.append(s)



    finaltext = ["The NAS system currently has %s free out of %s (%s)."
                    %(available, size, percent),
                '\n'.join(["_%s_     %s"%(a, b) for (a,b) in zip(wds, sizes)]) ]
except Exception as e:
    # Default message in case of failures
    finaltext = ['The control process has resulted in an error.', str(e)]


#===============================================================================
# Estimates elapsed time
seconds = datetime.now() - startTime
m, s = divmod(seconds.total_seconds(), 60)
h, m = divmod(m, 60)
elapsedtime = "%d:%02d:%02d" % (h, m, s)

#====================
# Sends the report to Slack
webhook_url = 'https://hooks.slack.com/services/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'

text, attachment = finaltext
payload = {"text": "%s Elapsed time: %s. (nas-monitor v0.1)"%(text, elapsedtime),
            "attachments": [{
              'fallback': 'Usage details',
              "text": attachment,
              "mrkdwn_in": ["text", "pretext"]
              }],
           "channel": "#general",
           "link_names": 1,
           "username": "nas-monitor",
           "icon_emoji": ":computer:"
          }

payload = json.dumps(payload).replace('"', '\\"').replace('\n', '\\n')

cmd = 'curl -H "Content-Type: application/json" --data "%s" %s'\
        %(payload, webhook_url)
print cmd

os.system(cmd)
