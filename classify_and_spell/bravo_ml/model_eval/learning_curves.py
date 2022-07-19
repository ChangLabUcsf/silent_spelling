
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_LC(results_dict, key, total=1, n_class=1, x = 'samples_per_class', 
           extra = '', n_LC_points=1):
    """
    Inputs: results dictionary
    Key: The key you want to be looking at (E.g. 'nato')
    total = the total number of samples 
    n_class = the number of classe
    x = what you want to call the x axis. The y axis is hard coded to acc rightnow. 
    Extra = a string for the plot title. 
    n_points = number of POINTS in the learning curve :D
    
    Outputs: 
    
    Makes a plot of the figure. Returns the dataframe of 'acc' and 'data_amount' or whatever you put in for x. 
    
    """
    cv_dict= {}

    for k, res in enumerate(results_dict[key]):
        if not k%n_LC_points in cv_dict: 
            cv_dict[k%n_LC_points] = []
        cv_dict[k%n_LC_points].append(res)
    import pandas as pd

    df = {}
    df[x] = []
    df['acc'] = []
    for k, v in cv_dict.items(): 
        df[x].extend([total*(k+1)/len(cv_dict)/n_class]*len(v))
        df['acc'].extend(v)
    df = pd.DataFrame(df)

    import seaborn as sns
    sns.lineplot(x=x, y='acc', data=df, label=extra)
    plt.title('%s %s' %(extra, key))
    return df