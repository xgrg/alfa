#! /usr/bin/env python
''' This generates an HTML table counting the different diskitems in an existing
Axon database.

Usage:
t = AxonInventory()
t.run('/home/data/repository/') # must be declared in BrainVisa databases
html = t.to_html(style='maxcdn')
'''

class AxonInventory(object):
    def __init__(self):
        pass

    def run(self, directory):
        import os.path as osp
        directory = osp.abspath(directory)
        from brainvisa import axon
        axon.initializeProcesses()
        import neuroHierarchy
        for direct, db in neuroHierarchy.databases._databases.items():
            if direct == directory:
               break
        if direct != directory:
            raise Exception('%s should be a valid Axon database.\nCurrent'\
                    'existing Axon databases : %s'\
                    %(directory, neuroHierarchy.databases._databases.keys()))

        items = [each for each in db.findDiskItems(**{'_type': 'Any Type'})]

        self.identified = {}
        self.index = []
        self.firstcol = 'subject'

        self.headers = set()

        for each in items:
            subject = each.get(self.firstcol)
            self.index.append(subject)
            self.identified.setdefault(subject, {})
            self.headers.add(each.type.name)
            fp = each.fullPath()
            self.identified[subject].setdefault(each.type.name, []).append(fp)

        self.index = sorted(list(set(self.index)))
        self.headers = sorted(list(self.headers))
        self.repository = directory

        self.count_table = []
        self.table = []

        for s in self.index:
          h = self.headers
          self.count_table.append([len(self.identified[s].get(e, [])) for e in h])
          self.table.append([self.identified[s].get(e, []) for e in h])

    def to_html(self, style="standalone"):
       if not hasattr(self, 'identified'):
          raise Exception('run Inventory.run() first')
       if style == 'standalone':
          html = '''<html><head></head><body><style>
               table.table, td, th { border: 1px solid darkgray;
               text-align:center; vertical-align:middle;
               }
               td.success { background-color:forestgreen}
               td.warning { background-color:goldenrod}
               td.danger {background-color:brown}
               </style>'''
       elif style == 'maxcdn':
          html = '''<html><head>
               <script src="https://code.jquery.com/jquery-1.12.0.min.js">
               </script>
               <!-- Latest compiled and minified CSS -->
               <link rel="stylesheet"
               href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css"
               integrity="sha384-1q8mTJOASx8j1Au+a5WDVnPi2lkFfwwEAa8hDDdjZlpLegxhjVME1fgjWPGmkzs7"
               crossorigin="anonymous">
               <!-- Optional theme -->
               <link rel="stylesheet"
               href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap-theme.min.css"
               integrity="sha384-fLW2N01lMqjakBkx3l/M9EahuwpSfeNvV63J5ezn3uZzapT0u7EYsXMjQV+0En5r"
               crossorigin="anonymous">
               <!-- Latest compiled and minified JavaScript -->
               <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/js/bootstrap.min.js"
               integrity="sha384-0mSbJDEHialfmuBBQP6A4Qrprq5OVfW37PRR3j5ELqxss1yVqOtnepnHVP9aJ7xS"
               crossorigin="anonymous"></script>
               </head><body>'''
       html += '<table class="table table-hover"><tr><th>%s</th>'%self.firstcol
       html += ''.join(['<th>%s</th>'%each for each in self.headers])
       html += '</tr>'
       colormap = ['danger', 'success', 'warning']
       for s, c in zip(self.index, self.count_table):
           items = self.identified[s]
           html += '<tr><td>%s</td>'%s
           html += ''.join(['<td title="%s" class="%s">%s</td>'
               %(' \n'.join([each[len(self.repository)+1:] for each in items.get(e1, [])]),
                                                                            colormap[min(2,e2)],
                                                                            e2) for e1, e2 in zip(self.headers, c)])
           html += '</tr>'
       html += '</table></body></html>'
       return html

if __name__ == '__main__':
    import argparse, textwrap
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
	    description=textwrap.dedent(\
            ''' This generates an HTML table counting the different diskitems in an existing
            Axon database.
	    '''))

    parser.add_argument("-i", dest='input', type=str, required=True, help='Path to the data repository')
    parser.add_argument("-o", dest='output', type=str, required=True, help='HTML file to be generated')
    args = parser.parse_args()

    import os.path as osp
    if not osp.isdir(args.input):
        raise Exception('%s should be an existing directory'%args.input)

    t = AxonInventory()
    t.run(args.input)
    w = open(args.output, 'w')
    w.write(t.to_html(style='maxcdn'))
    w.close()
    print 'Inventory table successfully created.'
