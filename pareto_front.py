import pandas as pd 
import copy
import numpy as np
 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
def verify_domination(pts,total_array):
    #TODO adapt to several elements
    flag_non_dominant=True
    #Simple iterationn
   
    for el in total_array:
        
        if pts[0]<el[0] and pts[1]<el[1]:
            flag_non_dominant=False
            break

    return flag_non_dominant



def main():
    sns.set_style("darkgrid")
    #sns.set_style({"xtick.bottom":True})
    plt.rcParams['figure.figsize'] = 17,7.27
    
    folder_path="data/pareto_front/"


    df_original=pd.read_csv(folder_path+"full_data_inverted_SAS.csv",index_col=None)
    df_v=df_original.copy()
    new_columns=df_original.columns
    df_new=pd.DataFrame([],columns=df_original.columns)

    #best_molecules=pd.DataFrame([])
    #Define the first and second parameters
    pn1="pIC50"
    pn2="SAS"

    


    max_mol=100
    rank_list=[]
    non_dominated=[]
    rank_n=1
 
    while df_new.shape[0]<max_mol:
        #Each cycle will add indexes to the list    

        data=df_v[[pn1,pn2]].to_numpy()

        for idx in range(data.shape[0]):
            if idx in non_dominated:
                continue
            data_point=data[idx]
            new_data = copy.deepcopy(data)
            
            new_data=np.delete(new_data,idx,axis=0)
            
            my_flag=verify_domination(data_point,new_data)
            if my_flag:
                non_dominated.append(idx)

        selected=df_v.iloc[non_dominated]
        df_v=df_v.drop(non_dominated)
        df_v=df_v.reset_index(drop=True)
     
        some_list=np.ones(len(non_dominated))*rank_n
        some_list=some_list.tolist()
        rank_list+=some_list
        rank_n+=1
        non_dominated=[]
        df_new=pd.concat([df_new,selected],ignore_index=True)

    #Plot the molecules
    df_new["Rank"]=rank_list
    data_p1=df_v[[pn1,pn2]]

    data_p2=df_new[[pn1,pn2,"Rank"]]
    fig,ax=plt.subplots()

    x_label=pn1
    y_label=pn2

    rank_list_u=data_p2["Rank"].unique().tolist()

    poly_data=pd.DataFrame([],columns=[x_label,y_label,"Rank"])
    total_x=[]
    total_y=[]
    total_rank=[]
    for el in rank_list_u:
        
        temp_data=data_p2.loc[data_p2["Rank"]==el]
        
        x1=temp_data[x_label].to_numpy()
        y1=temp_data[y_label].to_numpy()
        n_size=x1.shape[0]
        x_min=np.amin(x1)
        x_max=np.amax(x1)
        z = np.polyfit(x1, y1, 4)
        p = np.poly1d(z)
        xp = np.linspace(x_min, x_max, 100)
    
        some_list=np.ones(len(xp))*el
        some_list=some_list.tolist()
        total_rank+=some_list
        total_x+=xp.tolist()
        total_y+=p(xp).tolist()
    
   
    poly_data[x_label]=total_x
    poly_data[y_label]=total_y
    poly_data["Rank"]=total_rank
 


    ax=sns.scatterplot(data=data_p1,x=x_label,y=y_label)
    sns.scatterplot(data=data_p2,x=x_label,y=y_label,hue="Rank",palette="dark:salmon_r")
    sns.lineplot(data=data_p2,x=x_label,y=y_label,hue="Rank",palette="dark:salmon_r",legend=False)
    ##graph=sns.lmplot(data=data_p2,x=x_label,y=y_label,hue="Rank",palette="crest",order=3,ci=0)

    plt.ylabel("SAS Score")
    plt.savefig("experiment_pareto_front.png")

    plt.show()







if __name__=="__main__":
    main()













