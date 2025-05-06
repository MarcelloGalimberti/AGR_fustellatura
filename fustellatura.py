# Processo fustellatura
# env neuraplprophet conda


import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO
import warnings
import matplotlib.pyplot as plt
import plotly.express as px
warnings.filterwarnings('ignore')
import plotly.graph_objects as go

####### Impaginazione

st.set_page_config(layout="wide")

url_immagine = 'LOGO-Artigrafiche_Italia.png'#?raw=true' #LOGO-Artigrafiche_Italia.png

col_1, col_2 = st.columns([1, 5])

with col_1:
    st.image(url_immagine, width=200)

with col_2:
    st.title('Analisi performance fustellatura | 2024 - YTD')

st.subheader('Caricamento dati | db fustellatura.xlsx', divider='gray')

####### Caricamento dati

uploaded_db = st.file_uploader("Carica db fustellatura.xlsx)") # nome file da caricare
if not uploaded_db:
    st.stop()
db_raw=pd.read_excel(uploaded_db, parse_dates=True) #, skiprows=[0,1,2,3,4], dtype={'Seriale': str}
db_raw['Day'] = pd.to_datetime(db_raw['Day'], format='%d/%m/%Y').dt.date

# dal 2024-01-01
db_raw = db_raw[db_raw['Day'] >= pd.to_datetime('2024-01-01').date()]

#st.dataframe(db_raw)

######### Pre-processing

db_sorted = db_raw.sort_values(by=['Capo','Risorsa SAS', 'Day', 'Commessa'], ascending=[True, True, True, True])
db_sorted = db_sorted.reset_index(drop=True)

#st.dataframe(db_sorted)

df = db_sorted[['Capo', 'Risorsa SAS', 'Day', 'Commessa', 'Volume Prod. (E)', 'Ore Totali (A+B+C+H)', 'Numero Avviamenti Primari (G)',
                'Cliente Terzo']] # aggiunto Cliente Terzo
df = df.dropna(subset=['Volume Prod. (E)'])
df['Run'] = 0

#st.dataframe(df)

capoconto = df['Capo'].unique().tolist()
st.write('Numero di capiconti:', len(capoconto))


df_run = pd.DataFrame(columns=['Capo', 'Risorsa SAS', 'Day', 'Commessa', 'Volume Prod. (E)', 'Ore Totali (A+B+C+H)', 'Numero Avviamenti Primari (G)', 'Run',
                               'Cliente Terzo']) # aggiunto Cliente Terzo
df_pivot_complessivo = pd.DataFrame()
for capo in capoconto:
    df_capo = df[df['Capo'] == capo]
    risorse = df_capo['Risorsa SAS'].unique().tolist()
    for risorsa in risorse:
        df_risorsa = df_capo[df_capo['Risorsa SAS'] == risorsa]
        df_risorsa = df_risorsa.reset_index(drop=True)
        run_counter = 0
        for i in range (len(df_risorsa)):
            if i  == len(df_risorsa):
                break
            if df_risorsa.at[i, 'Numero Avviamenti Primari (G)'] == 1:
                run_counter += 1
                df_risorsa.at[i, 'Run'] = run_counter   
            else:
                df_risorsa.at[i, 'Run'] = run_counter
        df_pivot = df_risorsa.pivot_table(index=['Capo', 'Risorsa SAS', 'Run','Cliente Terzo'], values=['Day','Volume Prod. (E)', 'Ore Totali (A+B+C+H)'],
                                           aggfunc={'Day':'min', 'Volume Prod. (E)':'sum','Ore Totali (A+B+C+H)':'sum'
                                                    }).reset_index() # aggiunto Cliente Terzo
        df_pivot_complessivo = pd.concat([df_pivot_complessivo, df_pivot], ignore_index=True)
        df_pivot_complessivo = df_pivot_complessivo.reset_index(drop=True)
        df_run = pd.concat([df_run, df_risorsa], ignore_index=True)  
        df_run = df_run.reset_index(drop=True)


########## scarica excel

def to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Foglio1')
    return output.getvalue()

############ Analisi run 

df_analisi = df_pivot_complessivo.copy()
df_analisi['Run'] = df_analisi['Run'].astype(int)
df_analisi['VLL'] = df_analisi['Volume Prod. (E)'] / df_analisi['Ore Totali (A+B+C+H)']
#df_analisi['VLL'] = df_analisi['VLL'].replace([np.inf, -np.inf], 0)
df_analisi['VLL'] = df_analisi['VLL'].round(2)
df_analisi['VLL'] = df_analisi['VLL'].astype(float)

st.header('Analisi run complessivo', divider='gray')
st.dataframe(df_analisi)

###### scarica excel
# Crea il bottone per scaricare df_analisi
analisi = to_excel_bytes(df_analisi)
st.download_button(
    label="📥 Scarica Analisi Run Complessivo",
    data=analisi,
    file_name='Analisi_fustellatura.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)



st.subheader('Tree chart capiconti', divider='gray')
fig_treemap = px.treemap(
    df_analisi,
    path=['Capo'],        
    values='Volume Prod. (E)',       
    width=1200, height=600,  
    hover_data=['Cliente Terzo'],
    color='Capo',        
)
st.plotly_chart(fig_treemap, use_container_width=True)

st.subheader('Analisi run per capoconto', divider='gray')
codice = st.selectbox('Seleziona capoconto', options=capoconto, index=0)
st.write("Capoconto selezionato: ", codice)
df_codice = df_analisi[df_analisi['Capo'] == codice]

##### selezione periodo

date_range = st.slider('Seleziona periodo:', df_codice['Day'].min(), df_codice['Day'].max(), (df_codice['Day'].min(), df_codice['Day'].max()), format="DD/MM/YYYY")

df_codice = df_codice[(df_codice['Day'] >= date_range[0]) & (df_codice['Day'] <= date_range[1])]
df_codice = df_codice.reset_index(drop=True)

df_codice.sort_values(by=['Day'], ascending=True, inplace=True)
df_codice = df_codice.reset_index(drop=True)

df_codice['Prog'] = range(1, len(df_codice) + 1)
df_codice['Day_Prog'] = df_codice['Day'].astype(str) + '_' + df_codice['Prog'].astype(str)
df_codice['Anno'] = pd.to_datetime(df_codice['Day']).dt.year
df_codice['Anno'] = df_codice['Anno'].astype(str)
df_codice['Mese'] = pd.to_datetime(df_codice['Day']).dt.month
#
st.write('Analisi run per capoconto:', codice)
st.dataframe(df_codice)




fig_run = px.bar(
    df_codice,
    x='Day_Prog',
    y='VLL',
    color='Risorsa SAS',
    title=f'Run per Capoconto: {codice}',
    labels={'VLL': 'Velocità lordissima', 'Day_Prog': 'Run data_progressivo'},
    #template='plotly_dark'
)
fig_run.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': df_codice['Day_Prog']})

# linea di tendenza
z = np.polyfit(range(len(df_codice)), df_codice['VLL'], 1)  # retta di regressione: grado 1
trend = np.poly1d(z)(range(len(df_codice)))  # valori y della linea di tendenza

fig_run.add_trace(
    go.Scatter(
        x=df_codice['Day_Prog'],
        y=trend,
        mode='lines',
        name='Trend',
        line=dict(dash='dash', width=2)
    )
)
st.plotly_chart(fig_run, use_container_width=True)

# grafico a dispersione con linea di tendenza
# Scatter base con colore per 'Anno'
fig_scatter = px.scatter(
    df_codice,
    x='Volume Prod. (E)',
    y='VLL',
    color='Anno',
    hover_data=['Day_Prog','Cliente Terzo'],
    labels={
        'Volume Prod. (E)': 'Tiratura',
        'VLL': 'Velocità lordissima',
        'Anno': 'Anno'
    },
    title='Velocità lordissima vs Tiratura per anno',
)

# Fit lineare su tutti i dati
x_vals = df_codice['Volume Prod. (E)']
y_vals = df_codice['VLL']
coeffs = np.polyfit(x_vals, y_vals, deg=1)
trendline = np.poly1d(coeffs)(x_vals)

# Fit logaritmico: y = a * ln(x) + b
# Filtra per evitare log(0) o log(x < 0)
mask = x_vals > 0
x_log = np.log(x_vals[mask])
y_log = y_vals[mask]

coeffs_log = np.polyfit(x_log, y_log, deg=1)
a, b = coeffs_log

# Calcola valori per linea di tendenza
x_fit = np.linspace(x_vals[mask].min(), x_vals[mask].max(), 100)
y_fit = a * np.log(x_fit) + b

# Aggiungi linea logaritmica
fig_scatter.add_trace(
    go.Scatter(
        x=x_fit,
        y=y_fit,
        mode='lines',
        name='Trend Logaritmico',
        line=dict(dash='dot', color='black')
    )
)



# Aggiungi linea di tendenza al grafico
#fig_scatter.add_trace(
#    go.Scatter(
#        x=x_vals,
#        y=trendline,
#        mode='lines',
#        name='Trend Globale',
#        line=dict(dash='dash', color='black', width=2)
#    )
#)

st.plotly_chart(fig_scatter, use_container_width=True)

###### scarica excel
# Crea il bottone per scaricare capoconto
analisi_capoconto = to_excel_bytes(df_codice)
st.download_button(
    label=f"📥 Scarica Analisi Run Capoconto {codice} dal {date_range[0]} al {date_range[1]}",
    data=analisi_capoconto,
    file_name=f'Analisi_fustellatura_{codice}_{date_range[0]}_{date_range[1]}.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)


st.stop()


