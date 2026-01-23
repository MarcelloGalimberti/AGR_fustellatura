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


########## scarica excel

def to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Foglio1')
    return output.getvalue()



####### Impaginazione

st.set_page_config(layout="wide")

url_immagine = 'https://github.com/MarcelloGalimberti/AGR_fustellatura/blob/main/LOGO-Artigrafiche_Italia.png?raw=true'
#'LOGO-Artigrafiche_Italia.png'#?raw=true' #LOGO-Artigrafiche_Italia.png
#https://github.com/MarcelloGalimberti/AGR_fustellatura/blob/main/LOGO-Artigrafiche_Italia.png?raw=true

col_1, col_2 = st.columns([1, 5])

with col_1:
    st.image(url_immagine, width=200)

with col_2:
    st.title('Analisi performance fustellatura')

st.subheader('Caricamento dati | Lean Seletti.xlsx', divider='gray')

####### Caricamento dati

uploaded_db = st.file_uploader("Carica Lean Seletti.xlsx)") # nome file da caricare
if not uploaded_db:
    st.stop()

# crea funzine per cache dati
@st.cache_data
def load_data(uploaded_db):
    db_raw=pd.read_excel(uploaded_db, parse_dates=True)  
    db_raw['DATA_CHIUSURA'] = pd.to_datetime(db_raw['DATA_CHIUSURA'], format='%Y%m%d').dt.date
    return db_raw


db_raw = load_data(uploaded_db)

# Calcola velocità di run; controlla divisione per zero
# evita divisione per zero
db_raw['PRODUZIONE_PREV'] = db_raw['PRODUZIONE_PREV'].replace(0, np.nan)
db_raw['VEL_RUN'] = db_raw['QTA_PREVISTA'] / db_raw['PRODUZIONE_PREV']

st.write('Anteprima dati caricati con VEL_RUN:')
st.write('Numero di righe totali:', len(db_raw))
# st.dataframe(db_raw)

df_fustellatura = db_raw[db_raw['COD_REPARTO']=='FUST']
macchine_fustellatura = ['A154','A152','A149','A148','A155','A145'] # CAVRIAGO

# mantieni solo macchine fustellatura
df_fustellatura['COD_MACCHINA'] = df_fustellatura['COD_MACCHINA'].str.strip()
df_fustellatura = df_fustellatura[df_fustellatura['COD_MACCHINA'].isin(macchine_fustellatura)]
#df_fustellatura = df_fustellatura.reset_index(drop=True)
# st.write('Numero di righe fustellatura prima del filtro QTA_PREVISTA >= 1000:', len(df_fustellatura))
# # calcola la percentuale di righe con QTA_PREVISTA < 1000
# percentuale_bassa_qta = (len(df_fustellatura[df_fustellatura['QTA_PREVISTA'] < 1000]) / len(df_fustellatura)) * 100
# st.write(f'Percentuale di righe con Quantità Prevista < 1000: {percentuale_bassa_qta:.2f}%')

# filtra per QTA_PREVISTA >= 1000 e QTA_PRODOTTA >= 1000
df_fustellatura = df_fustellatura[df_fustellatura['QTA_PREVISTA'] >= 1000]
df_fustellatura = df_fustellatura[df_fustellatura['QTA_PRODOTTA'] >= 1000]

df_fustellatura.sort_values(by=['DATA_CHIUSURA'], ascending=True, inplace=True)
df_fustellatura = df_fustellatura.reset_index(drop=True)
st.write('Numero di righe fustellatura:', len(df_fustellatura))

# se VEL_RUN >= 9500 imposta a NaN
df_fustellatura.loc[df_fustellatura['VEL_RUN'] >= 9500, 'VEL_RUN'] = np.nan

st.dataframe(df_fustellatura)

# Crea pulsante per scaricare dati fustellatura
fustellatura_excel = to_excel_bytes(df_fustellatura)
st.download_button(
    label="📥 Scarica dati fustellatura filtrati",
    data=fustellatura_excel,
    file_name='Dati_fustellatura_filtrati.xlsx',
    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
)

# st.write('Statistiche VEL_RUN fustellatura:')
# st.write(df_fustellatura['VEL_RUN'].describe())

# st.write('Distribuzione VEL_RUN fustellatura:')
# fig = px.histogram(df_fustellatura, x='VEL_RUN', nbins=30, title='Distribuzione VEL_RUN Fustellatura', color_discrete_sequence=['skyblue'])
# fig.update_traces(marker_line_color='black', marker_line_width=1)
# fig.update_layout(xaxis_title='VEL_RUN', yaxis_title='Frequenza')
# st.plotly_chart(fig, use_container_width=True)

# Statistiche VEL_MEDIA_PREV
# st.write('Statistiche VEL_MEDIA_PREV fustellatura:')
# st.write(df_fustellatura['VEL_MEDIA_PREV'].describe())

# st.write('Distribuzione VEL_MEDIA_PREV fustellatura:')
# fig1 = px.histogram(df_fustellatura, x='VEL_MEDIA_PREV', nbins=30, title='Distribuzione VEL_MEDIA_PREV Fustellatura', color_discrete_sequence=['orange'])
# fig1.update_traces(marker_line_color='black', marker_line_width=1)
# fig1.update_layout(xaxis_title='VEL_MEDIA_PREV', yaxis_title='Frequenza')
# st.plotly_chart(fig1, use_container_width=True)

# # Statistiche VEL_MEDIA_CONS
# st.write('Statistiche VEL_MEDIA_CONS fustellatura:')

# st.write(df_fustellatura['VEL_MEDIA_CONS'].describe())
# st.write('Distribuzione VEL_MEDIA_CONS fustellatura:')
# fig3 = px.histogram(df_fustellatura, x='VEL_MEDIA_CONS', nbins=30, title='Distribuzione VEL_MEDIA_CONS Fustellatura', color_discrete_sequence=['violet'])
# fig3.update_traces(marker_line_color='black', marker_line_width=1)
# fig3.update_layout(xaxis_title='VEL_MEDIA_CONS', yaxis_title='Frequenza')
# st.plotly_chart(fig3, use_container_width=True)


# st.write('Statistiche Quantità Prevista fustellatura:')
# st.write(df_fustellatura['QTA_PREVISTA'].describe())

# st.write('Distribuzione Quantità Prevista fustellatura:')
# fig2 = px.histogram(df_fustellatura, x='QTA_PREVISTA', nbins=30, title='Distribuzione Quantità Prevista Fustellatura', color_discrete_sequence=['lightgreen'])
# fig2.update_traces(marker_line_color='black', marker_line_width=1)
# fig2.update_layout(xaxis_title='Quantità Prevista', yaxis_title='Frequenza')
# st.plotly_chart(fig2, use_container_width=True)


st.subheader('Commesse con possibili anomalie di preventivo', divider='gray')
df_anomalie_preventivo = df_fustellatura[df_fustellatura['VEL_MEDIA_PREV'] >= 9500]
st.write('Numero di righe con possibili anomalie di preventivo (VEL_MEDIA_PREV >= 9500):', len(df_anomalie_preventivo))
st.dataframe(df_anomalie_preventivo)

st.subheader('Commesse con possibili anomalie di consuntivo', divider='gray')
df_anomalie_consuntivo = df_fustellatura[df_fustellatura['VEL_MEDIA_CONS'] >= 9500]
st.write('Numero di righe con possibili anomalie di consuntivo (VEL_MEDIA_CONS >= 9500):', len(df_anomalie_consuntivo))
st.dataframe(df_anomalie_consuntivo)

st.subheader('Dati da analizzare | Fustellatura', divider='gray')
df_analisi = df_fustellatura.copy()
# elimina righe con VEL_MEDIA_PREV >= 9500 o VEL_MEDIA_CONS >= 9500
df_analisi = df_analisi[df_analisi['VEL_MEDIA_PREV'] < 9500]
df_analisi = df_analisi[df_analisi['VEL_MEDIA_CONS'] < 9500]
st.write('Numero di righe per analisi dopo aver rimosso anomalie:', len(df_analisi))

colonne_rilevanti = ['DATA_CHIUSURA','PROGR_COMMESSA','CAPOCONTO','RAGIONE_SOC','COD_MACCHINA','DES_MACCHINA',
                     'QTA_PREVISTA','QTA_PRODOTTA','PRODUZIONE_PREV','PRODUZIONE_CONS','ATTESE_PREV','ATTESE_CONS',
                     'AVVIAMENTO_PREV','AVVIAMENTO_CONS',
                     'VEL_RUN','VEL_MEDIA_PREV','VEL_MEDIA_CONS','RESA_FOGLIO']

df_analisi = df_analisi[colonne_rilevanti]
df_analisi = df_analisi.reset_index(drop=True)
df_analisi['VEL_MEDIA_CONS-PREV'] = df_analisi['VEL_MEDIA_CONS'] - df_analisi['VEL_MEDIA_PREV']

st.write('Dati da analizzare:')

st.dataframe(df_analisi)

# Statistica VEL_MEDIA_CONS-PREV
st.write('Statistiche VEL_MEDIA_CONS-PREV fustellatura:')
st.write(df_analisi['VEL_MEDIA_CONS-PREV'].describe())

#st.write('Distribuzione VEL_MEDIA_CONS-PREV fustellatura:')
fig4 = px.histogram(df_analisi, x='VEL_MEDIA_CONS-PREV', nbins=30, title='Distribuzione VEL_MEDIA_CONS-PREV Fustellatura', color_discrete_sequence=['lightcoral'])
fig4.update_traces(marker_line_color='black', marker_line_width=1)
fig4.update_layout(xaxis_title='VEL_MEDIA_CONS-PREV', yaxis_title='Frequenza')
st.plotly_chart(fig4, use_container_width=True)


# Inserisci slider per filtro data
data_min = df_analisi['DATA_CHIUSURA'].min()
data_max = df_analisi['DATA_CHIUSURA'].max()
data_selezionata = st.slider('Seleziona intervallo di date per l\'analisi:', min_value=data_min, max_value=data_max, value=(data_min, data_max))

df_analisi_filtrato = df_analisi[(df_analisi['DATA_CHIUSURA'] >= data_selezionata[0]) & (df_analisi['DATA_CHIUSURA'] <= data_selezionata[1])]

st.write('Dati filtrati per intervallo di date selezionato:')
st.dataframe(df_analisi_filtrato)

# Calcolo Z-score
if not df_analisi_filtrato.empty:
    mean_val = df_analisi_filtrato['VEL_MEDIA_CONS-PREV'].mean()
    std_val = df_analisi_filtrato['VEL_MEDIA_CONS-PREV'].std()
    
    if std_val != 0:
        df_analisi_filtrato['Z_SCORE'] = (df_analisi_filtrato['VEL_MEDIA_CONS-PREV'] - mean_val) / std_val
    else:
        df_analisi_filtrato['Z_SCORE'] = 0
        
    df_analisi_filtrato['Stato'] = df_analisi_filtrato['Z_SCORE'].apply(
        lambda x: 'Positivo (Z > 2)' if x > 2 else ('Negativo (Z < -2)' if x < -2 else 'Normale')
    )

    st.write('Analisi Z-score (Verde > 2, Rosso < -2):')
    fig_zscore = px.scatter(
        df_analisi_filtrato,
        x='DATA_CHIUSURA',
        y='Z_SCORE',
        color='Stato',
        color_discrete_map={'Positivo (Z > 2)': 'green', 'Negativo (Z < -2)': 'red', 'Normale': 'blue'},
        title='Andamento Z-score VEL_MEDIA_CONS-PREV',
        hover_data=['PROGR_COMMESSA', 'CAPOCONTO', 'VEL_MEDIA_CONS-PREV']
    )
    fig_zscore.add_hline(y=2, line_dash="dash", line_color="red", annotation_text="Soglia Z=2")
    fig_zscore.add_hline(y=-2, line_dash="dash", line_color="red", annotation_text="Soglia Z=-2")
    st.plotly_chart(fig_zscore, use_container_width=True)


st.stop()


# dal 2024-01-01
db_raw = db_raw[db_raw['DATA_CHIUSURA'] >= pd.to_datetime('2024-01-01').date()]

#st.dataframe(db_raw)

######### Pre-processing

db_sorted = db_raw.sort_values(by=['Capoconto','Risorsa SAS', 'Day', 'Commessa'], ascending=[True, True, True, True])
db_sorted = db_sorted.reset_index(drop=True)

#st.dataframe(db_sorted)

df = db_sorted[['Capoconto', 'Risorsa SAS', 'Day', 'Commessa', 'Volume Prod. (E)', 'Ore Totali (A+B+C+H)', 'Numero Avviamenti Primari (G)',
                'Cliente Terzo']] # aggiunto Cliente Terzo
df = df.dropna(subset=['Volume Prod. (E)'])
df['Run'] = 0

#st.dataframe(df)

capoconto = df['Capoconto'].unique().tolist()
st.write('Numero di capiconti:', len(capoconto))


df_run = pd.DataFrame(columns=['Capoconto', 'Risorsa SAS', 'Day', 'Commessa', 'Volume Prod. (E)', 'Ore Totali (A+B+C+H)', 'Numero Avviamenti Primari (G)', 'Run',
                               'Cliente Terzo']) # aggiunto Cliente Terzo
df_pivot_complessivo = pd.DataFrame()
for capo in capoconto:
    df_capo = df[df['Capoconto'] == capo]
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
        df_pivot = df_risorsa.pivot_table(index=['Capoconto', 'Risorsa SAS', 'Run','Cliente Terzo'], values=['Day','Volume Prod. (E)', 'Ore Totali (A+B+C+H)'],
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
    path=['Capoconto'],        
    values='Volume Prod. (E)',       
    width=1200, height=600,  
    hover_data=['Cliente Terzo'],
    color='Capoconto',        
)
st.plotly_chart(fig_treemap, use_container_width=True)

st.subheader('Analisi run per capoconto', divider='gray')
codice = st.selectbox('Seleziona capoconto', options=capoconto, index=0)
st.write("Capoconto selezionato: ", codice)
df_codice = df_analisi[df_analisi['Capoconto'] == codice]

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


