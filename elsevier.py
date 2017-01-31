import pymssql,gzip,codecs
from tqdm import tqdm as tq

ddir = 'S:/UsersData_NoExpiration/jjl2228/wos-text-dynamics-data/elsevier/raw/'
total_lines=7775842

### Server setup
server,user,password = [line.strip() for line in open('server_credentials.txt')]
conn = pymssql.connect(server, user, password, "tempdb")
cursor = conn.cursor()

with gzip.open('data/SD_WoS_id_match.txt.gz') as id_file:
    for line in tq(id_file,total=total_lines):
        el_id,wos_id = line.strip().split()
        cursor.execute("SELECT PaperContent  FROM [Papers].[dbo].[Papers] where FileID={}".format(el_id))
        result = cursor.fetchone()[0]
        with codecs.open(ddir+wos_id,'w',encoding='utf8') as out:
            out.write(result)

cursor.close()
conn.close()
