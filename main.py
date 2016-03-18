import datetime
import time
from Tkinter import *
import tkMessageBox
import MySQLdb
import pandas
from sshtunnel import SSHTunnelForwarder
from sklearn.externals import joblib
import dpro

_inx = [u'Accel_X|slope_diff', u'Accel_X|std_diff', u'Accel_Y|slope',
        u'Accel_X.lead2|std_ac_xyz2', u'Accel_X|max',
        u'Accel_X.lead2|mean_diff', u'Accel_X.lag2|slope',
        u'Accel_X.lag2|mean_diff', u'Accel_X|slope', u'Accel_X|mean_diff']

with SSHTunnelForwarder(('murphy.wot.eecs.northwestern.edu', 22), ssh_password='6966lanyumi!', ssh_username='xzt387',
                        remote_bind_address=('127.0.0.1', 3306)) as server:
    con = MySQLdb.connect(host='127.0.0.1', port=server.local_bind_port, user='mhealth', passwd='mhealth',
                          db='mhealthplay')
    useless = pandas.read_sql_query('select * from SmokingRing; delete from SmokingRing;', con=con)
    time.sleep(10)
    try:
        while True:
            sk = pandas.read_sql_query('select * from SmokingRing;', con=con)
            # process the raw signals
            dat = dpro.process(sk)
            count = len(sk)/48
            query = 'select * from SmokingRing; delete from SmokingRing ORDER BY ID ASC limit %s;' % count
            heihei = pandas.read_sql_query(query, con=con)
            # remove all rows with null values
            dat = dat.dropna()
            # load classifier options:
            # clf_btree_refined
            # clf_adaboost.pkl
            # clf_rf.pkl
            # clf_svm.pkl
            clf = joblib.load('model/clf_rf.pkl')
            # print dir(clf)
            # make predictions
            pre = clf.predict(dat[_inx])
            print pre
            if 1 in pre:
                now = datetime.datetime.now()
                print now.strftime("%Y-%m-%d %H:%M:%S"), " Smoking detected!"
                time.sleep(1)
                window = Tk()
                window.wm_withdraw()
                window.geometry("1x1+"+str(window.winfo_screenwidth()/2)+"+"+str(window.winfo_screenheight()/2))
                tkMessageBox.showinfo(title="WARNING!", message="SMOKING DETECTED!")
                break
            now = datetime.datetime.now()
            print now.strftime("%Y-%m-%d %H:%M:%S"), " Not smoking."
            time.sleep(5)
    except KeyboardInterrupt:
        pass




