import plotly.graph_objects as go
import plotly.express as px

def plot_duval_triangle(df):
    """
    Plots the Duval Triangle 1 using plotly ternary plot.
    """
    df_plot = df.copy()
    sum_gases = df_plot['CH4'] + df_plot['C2H4'] + df_plot['C2H2']
    
    # Avoid division by zero
    sum_gases = sum_gases.replace(0, 1)
    
    df_plot['%CH4'] = (df_plot['CH4'] / sum_gases) * 100
    df_plot['%C2H4'] = (df_plot['C2H4'] / sum_gases) * 100
    df_plot['%C2H2'] = (df_plot['C2H2'] / sum_gases) * 100
    
    # Ternary Plot
    fig = px.scatter_ternary(
        df_plot, 
        a="%CH4", 
        b="%C2H4", 
        c="%C2H2", 
        color="Fault_Type",
        title="Duval Triangle 1 Analysis",
        labels={"%CH4": "%CH4", "%C2H4": "%C2H4", "%C2H2": "%C2H2"}
    )
    
    return fig

def duval_1_classification(ch4, c2h4, c2h2):
    total = ch4 + c2h4 + c2h2
    if total == 0: return "Normal"
    
    p_ch4 = (ch4 / total) * 100
    p_c2h4 = (c2h4 / total) * 100
    p_c2h2 = (c2h2 / total) * 100
    
    if p_ch4 >= 98:
        return "PD"
    elif p_c2h2 < 4 and p_c2h4 < 20:
        return "T1"
    elif p_c2h2 < 4 and 20 <= p_c2h4 < 50:
        return "T2"
    elif p_c2h2 < 15 and p_c2h4 >= 50:
        return "T3"
    elif p_c2h2 >= 13 and p_c2h4 < 23:
        return "D1"
    elif p_c2h2 >= 13 and p_c2h4 >= 23:
        return "D2"
    else:
        return "DT" # Mixed
