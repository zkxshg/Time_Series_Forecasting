import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('.//323//e022fa771e.csv')
all_points=[]
for num in range(df.shape[0]):
    onekeypoint=[]
    for i in range(17):     
        site=df.iloc[num,i]
        onekeypoint.append([float(site.split(',')[0]),float(site.split(',')[1])])
    x=[]
    y=[]
    for i,j in onekeypoint:
        x.append(i)
        y.append(j)
    
    plt.scatter(x, y)
    plt.show()
    
    # all_points.append(onekeypoint)
# X=[(172.58431595431966,47.25410568882042),(170.94070489833533,44.60222639548226),(174.7705841985507,45.038778201435164),
# (166.9291017098686,45.80693695866606),(178.7821044921875,46.33918624048981),(164.41899099676183,57.654533155967044),
# (182.53928502656566,57.05208116374024),(150.55439779792036,62.42023061218876),(196.9332722408671,58.194546920193034),
# (142.79970619616137,55.85929824552784),(203.56169942015373,56.29192636286467),(167.09619926590796,93.30083896936304),
# (178.6018819214113,93.95756865411215),(173.4635169414927,119.819889797771),(174.9409422845668,119.619118519712),
# (176.88595894212935,141.48393780652665),(172.41432628976986,141.70530384455168)]
