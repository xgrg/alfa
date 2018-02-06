#!/usr/bin/env python

import argparse
import logging as log
import os.path as osp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Updates an Axon database'\
        'without loading the GUI')
    parser.add_argument('database', help='Database to update')
    opts = parser.parse_args()

    from brainvisa import axon, processes

    axon.initializeProcesses()
    import neuroHierarchy
    found_databases = neuroHierarchy.databases._databases.keys()
    log.basicConfig(level=log.INFO)
    database = osp.abspath(opts.database)
    if not database in found_databases:
        log.error('Database \'%s\' not found among %s'%(database, found_databases))
    log.info('Updating %s'%database)
    db = neuroHierarchy.databases.database(database)
    db.update(context=processes.defaultContext())
    log.info('Successfully updated.')




