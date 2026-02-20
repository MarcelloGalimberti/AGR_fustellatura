# Elaborazione file OEE Board
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
from plotly.subplots import make_subplots
from datetime import timedelta

########## scarica excel

def to_excel_bytes(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Foglio1')
    return output.getvalue()



####### Impaginazione

st.set_page_config(layout="wide")

url_immagine = 'https://github.com/MarcelloGalimberti/AGR_fustellatura/blob/main/LOGO-Artigrafiche_Italia.png?raw=true'

col_1, col_2 = st.columns([1, 5])

with col_1:
    st.image(url_immagine, width=200)

with col_2:
    st.title('Dashboard di Produzione v3')

####### Caricamento dati

uploaded_db = st.sidebar.file_uploader("Carica Statist9 analisi lavorazioni db.xlsx)") # nome file da caricare
if not uploaded_db:
    st.warning("⚠️ Carica il file 'Statist9 analisi lavorazioni db.xlsx' nella sidebar per procedere.")
    st.stop()

# uploaded_abbinamento = st.sidebar.file_uploader("Carica abbinamento gruppo.xlsx)") # nome file da caricare
# if not uploaded_abbinamento:
#     st.warning("⚠️ Carica il file 'abbinamento gruppo.xlsx' nella sidebar per procedere.")
#     st.stop()

uploaded_lean = st.sidebar.file_uploader("Carica lean.xlsx)") # nome file da caricare
if not uploaded_lean:
    st.warning("⚠️ Carica il file 'lean.xlsx' nella sidebar per procedere.")
    st.stop()

uploaded_budget = "bgt_2026_volumi.xlsx"
uploaded_abbinamento = "Abbinamento_Gruppo.xlsx"

@st.cache_data
def load_budget(uploaded_budget):
    budget=pd.read_excel(uploaded_budget, parse_dates=True)
    return budget

budget = load_budget(uploaded_budget)
dizionario_budget = {'Stampa':'STAM','Fustellatura':'FUST','Piega_incolla':'INCO'}
# nel budget rinomina le righe della colonna Gruppo_risorse in base al dizionario
budget['Gruppo_risorse'] = budget['Gruppo_risorse'].replace(dizionario_budget)

# crea funzione per cache dati
@st.cache_data
def load_data(uploaded_db):
    db_raw=pd.read_excel(uploaded_db, parse_dates=True)
    return db_raw

@st.cache_data
def load_abbinamento(uploaded_abbinamento):
    abbinamento=pd.read_excel(uploaded_abbinamento, parse_dates=True)
    return abbinamento

@st.cache_data
def load_lean(uploaded_lean):
    lean=pd.read_excel(uploaded_lean, parse_dates=True)
    lean['DATA_CHIUSURA'] = pd.to_datetime(lean['DATA_CHIUSURA'], format='%Y%m%d').dt.date
    return lean

db_raw = load_data(uploaded_db)
abbinamento = load_abbinamento(uploaded_abbinamento)
lean = load_lean(uploaded_lean)

# Calcola velocità di run; controlla divisione per zero
# evita divisione per zero
lean['PRODUZIONE_PREV'] = lean['PRODUZIONE_PREV'].replace(0, np.nan)
lean['COD_REPARTO'] = lean['COD_REPARTO'].str.strip()
lean['VEL_RUN'] = lean['QTA_PREVISTA'] / lean['PRODUZIONE_PREV']
lean['Risorsa SAS'] = lean['COD_MACCHINA'].str.strip()+ " " + lean['DES_MACCHINA'].str.strip()
colonne_lean_to_remove = ['LETT_COMMESSA','LOTTO','FOGLIO','FOGLIO_ST','SEQP','SEGF','PRRG',
                            'COD_OPERATORE','DES_OPERATORE','VEL_MEDIA_CAPOCONTO_18M',
                            'VEL_MEDIA_CAPOCONTO_MM_18M','VEL_MEDIA_CONS_MM','VEL_MEDIA_PREV_MM',
                            'ATTESE_MEDIA_CAPOCONTO_18M']
lean = lean.drop(colonne_lean_to_remove, axis=1)
lean['VEL_RUN'] = lean['VEL_RUN'].round(2)
lean['ID_COMMESSA'] = lean['ANNO_COMMESSA'].astype(str) + '-' + lean['PROGR_COMMESSA'].astype(str)
macchine_obsolete = ['A150 KMN FINEST. 105/2','A169 VEGA 80_2','A168 VEGA 80_1','A160 CTPACK 1','A163 BST ALPINA 75 CTPK_3',
                     'A115 EASY PRESS','A112 PREPARAZIONE LASTRE  N°','TAGLI TAGLIERINA']
lean = lean[~lean['Risorsa SAS'].isin(macchine_obsolete)]
lean = lean.merge(abbinamento[['Risorsa SAS','Gruppo','Plant']], on='Risorsa SAS', how='left')


tab1, tab2 = st.tabs(['**Dati mensili di Produzione**', '**Performance per periodo**'])



with tab2:
    st.write('Anteprima dati caricati con VEL_RUN:')
    st.write('Numero di righe totali:', len(lean))
    st.dataframe(lean)


    with st.expander('Dettaglio dati'):
        st.write('Data chiusura minima:', lean['DATA_CHIUSURA'].min())
        st.write('Data chiusura massima:', lean['DATA_CHIUSURA'].max())
        st.write('Numero di commesse:', lean['ID_COMMESSA'].nunique())
        st.write('Numero capoconti:', lean['CAPOCONTO'].nunique())
        st.write('Numero Clienti:', lean['COD_CLIENTE'].nunique())
        st.write('Numero settori merceologici:', lean['DESC_SET_MERC_VENDITA'].nunique())
        st.write('Numero segmenti di business:', lean['DESC_SEG_BUSINESS'].nunique())
        st.write('Numero settori di business:', lean['DESC_SEG_BUSINESS'].nunique())
        st.write('Numero di Risorse SAS:', lean['Risorsa SAS'].nunique())
        st.write('Numero di reparti:', lean['COD_REPARTO'].nunique())

    # st.write('Identificazione limiti per eliminazione dati non validi:')
    # # per ogni Risorsa SAS calcola le statistiche descrittive di VEL_RUN e mettile in un dataframe
    # stats_lean = lean.groupby('Risorsa SAS')['VEL_RUN'].describe()
    # st.dataframe(stats_lean)


    lean_filtered = lean[(lean['QTA_PREVISTA'] > lean['QTA_PREVISTA'].quantile(0.025)) & (lean['QTA_PRODOTTA'] > lean['QTA_PRODOTTA'].quantile(0.025))]
    lean_filtered.reset_index(drop=True, inplace=True)
    # st.write('Numero di righe totali:', len(lean_filtered))
    # st.dataframe(lean_filtered)

    # inserisce colonne per analisi
    lean_filtered['VEL_MEDIA_CONS-PREV'] = lean_filtered['VEL_MEDIA_CONS'] - lean_filtered['VEL_MEDIA_PREV']
    lean_filtered['VEL_MEDIA_CONS-PREV'] = lean_filtered['VEL_MEDIA_CONS-PREV'].round(2)
    lean_filtered['Scostamento VEL_MEDIA %'] = lean_filtered['VEL_MEDIA_CONS-PREV'] / lean_filtered['VEL_MEDIA_PREV']
    lean_filtered['Scostamento VEL_MEDIA %'] = lean_filtered['Scostamento VEL_MEDIA %'].round(2)
    lean_filtered['Scostamento Label'] = lean_filtered['Scostamento VEL_MEDIA %'].map('{:.2%}'.format)

    
    # Ordine reparti
    lista_reparti = lean_filtered['COD_REPARTO'].unique()
    sequenza_reparti = ['STAM','ACCO','FUST','MVAR','INCO','FINE']
    # ordina lista_reparti in base a sequenza_reparti
    lista_reparti = [x for x in sequenza_reparti if x in lista_reparti]
    
    
    # Selezione plant con st.radio
    lista_plants = lean_filtered['Plant'].unique()
    lista_plants.sort()
    plant = st.radio('**Plant**', lista_plants, horizontal=True)
    lean_filtered = lean_filtered[lean_filtered['Plant'] == plant]
    
    # Selezione del periodo di analisi con date_input
    min_date = lean_filtered['DATA_CHIUSURA'].min()
    max_date = lean_filtered['DATA_CHIUSURA'].max()
    
    # Calcola valori di default: data_fine = max_date -1 giorno, data_inizio = data_fine -1 settimana
    default_data_fine = max_date - timedelta(days=1)
    default_data_inizio = default_data_fine - timedelta(weeks=1)
    
    col_data_inizio, col_data_fine = st.columns(2)
    with col_data_inizio:
        data_inizio = st.date_input('Data inizio', value=default_data_inizio, min_value=min_date, max_value=max_date)
    with col_data_fine:
        data_fine = st.date_input('Data fine', value=default_data_fine, min_value=min_date, max_value=max_date)
    
    lean_filtered = lean_filtered[(lean_filtered['DATA_CHIUSURA'] >= data_inizio) & (lean_filtered['DATA_CHIUSURA'] <= data_fine)]
    st.write('Numero di commesse:', len(lean_filtered['ID_COMMESSA'].unique()))
    st.write('Giorni produttivi:', len(lean_filtered['DATA_CHIUSURA'].unique()))
    st.write('Media commesse/giorno:', round(len(lean_filtered['ID_COMMESSA'].unique())/len(lean_filtered['DATA_CHIUSURA'].unique()),2))
    st.write('Numero capoconti:', len(lean_filtered['CAPOCONTO'].unique()))
    with st.expander('Dati periodo selezionato'):
        st.dataframe(lean_filtered)
    
    st.subheader('Analisi performance | Scostamento % velocità media consuntiva vs prevista', divider='grey')
    
    # ordina i reparti in base alla sequenza
    reparti_periodo = lean_filtered['COD_REPARTO'].unique()
    reparti_periodo = [x for x in sequenza_reparti if x in reparti_periodo]
    #st.write('Reparti nel periodo:', reparti_periodo)

    lean_filtered['COD_REPARTO'] = pd.Categorical(lean_filtered['COD_REPARTO'], categories=sequenza_reparti, ordered=True)
    lean_filtered_sorted = lean_filtered.sort_values(by=['COD_REPARTO', 'Gruppo', 'DATA_CHIUSURA'])
    lean_filtered_sorted['COD_REPARTO'] = lean_filtered_sorted['COD_REPARTO'].astype(str)
    #st.dataframe(lean_filtered_sorted)

    gruppi_periodo = lean_filtered['Gruppo'].unique()
    #st.write('Gruppi nel periodo:', gruppi_periodo)
    
    def grafico_scostamenti(df):
        fig_scostamenti = px.scatter(
            df, 
            x='DATA_CHIUSURA', 
            y='Scostamento VEL_MEDIA %', 
            size='QTA_PRODOTTA',
            color='Risorsa SAS', 
            title='Scostamenti VEL_MEDIA %'
        )
        fig_scostamenti.update_traces(marker=dict(#size=12,
                                     line=dict(width=2,
                                     color='DarkSlateGrey')),
                                  selector=dict(mode='markers'))
        return fig_scostamenti
    

    def grafico_scostamenti_z_score(df):
        fig_scostamenti = px.scatter(
            df, 
            x='DATA_CHIUSURA', 
            y='z_score_scostamento', 
            size='QTA_PRODOTTA',
            color='Risorsa SAS', 
            title='Scostamenti VEL_MEDIA % | Z-Score',
            hover_data=['VEL_MEDIA_PREV', 'VEL_MEDIA_CONS', 'ID_COMMESSA', 'CAPOCONTO', 'RAGIONE_SOC', 'DESC_SEG_BUSINESS'],
            height=600

        )
        fig_scostamenti.update_traces(marker=dict(#size=12,
                                     line=dict(width=2,
                                     color='DarkSlateGrey')),
                                  selector=dict(mode='markers'))
        
        # Add horizontal lines
        fig_scostamenti.add_hline(y=2, line_dash="dash", line_color="orange", annotation_text="+2 SD")
        fig_scostamenti.add_hline(y=3, line_dash="dash", line_color="red", annotation_text="+3 SD")
        fig_scostamenti.add_hline(y=-2, line_dash="dash", line_color="orange", annotation_text="-2 SD")
        fig_scostamenti.add_hline(y=-3, line_dash="dash", line_color="red", annotation_text="-3 SD")
        
        fig_scostamenti.update_layout(showlegend=True) # True
        return fig_scostamenti

    def boxplot_scostamenti(df):
        fig_boxplot = px.box(
            df, 
            x='Risorsa SAS', 
            y='z_score_scostamento', 
            color='Risorsa SAS', 
            title='Scostamenti VEL_MEDIA % | z-score',
            height=600
        )
        
        # Add horizontal lines
        fig_boxplot.add_hline(y=2, line_dash="dash", line_color="orange", annotation_text="+2 SD")
        fig_boxplot.add_hline(y=3, line_dash="dash", line_color="red", annotation_text="+3 SD")
        fig_boxplot.add_hline(y=-2, line_dash="dash", line_color="orange", annotation_text="-2 SD")
        fig_boxplot.add_hline(y=-3, line_dash="dash", line_color="red", annotation_text="-3 SD")
        
        return fig_boxplot
    
    
    
    # UI Control for Outlier Removal
    col_toggle, col_robust = st.columns(2)
    with col_toggle:
        remove_outliers = st.toggle("Rimuovi outlier Z-Score")
    with col_robust:
        use_robust = st.toggle("Usa Z-Score Robusto (Mediana/MAD)")
        
    if remove_outliers:
        z_threshold = st.slider("Soglia Z-Score", min_value=3.0, max_value=10.0, value=5.0, step=0.1)

    # Funzione per calcolo Z-Score
    def calculate_z_score_series(series, robust=False):
        # Return zeros if series is empty or all NaNs to avoid errors
        if series.dropna().empty:
             return pd.Series(np.nan, index=series.index)

        if robust:
            median = series.median()
            # MAD = Median Absolute Deviation
            mad = (series - median).abs().median()
            if mad == 0:
                return pd.Series(0, index=series.index)
            # Fattore 0.6745 per rendere MAD comparabile a STD su dist. normale
            return 0.6745 * (series - median) / mad
        else:
            std = series.std()
            if std == 0 or np.isnan(std):
                return pd.Series(0, index=series.index)
            return (series - series.mean()) / std

    
    for reparto in reparti_periodo:
        
        st.write(f'**Reparto:    {reparto}**')
        df_reparto = lean_filtered[lean_filtered['COD_REPARTO'] == reparto]
        gruppi_di_reparto = df_reparto['Gruppo'].unique()
        df_reparto['Gruppo'] = pd.Categorical(df_reparto['Gruppo'], categories=gruppi_di_reparto, ordered=True)
        df_reparto_sorted = df_reparto.sort_values(by=['Gruppo', 'DATA_CHIUSURA'])
        df_reparto_sorted['Gruppo'] = df_reparto_sorted['Gruppo'].astype(str)

        for gruppi in gruppi_di_reparto:
            st.write(f'Gruppo: {gruppi}')
            df_gruppi = df_reparto_sorted[df_reparto_sorted['Gruppo'] == gruppi]
            
            # Calcolo Z-Score iniziale (Standard o Robusto)
            df_gruppi['z_score_scostamento'] = calculate_z_score_series(df_gruppi['Scostamento VEL_MEDIA %'], robust=use_robust)
            
            # Logica rimozione outlier
            if remove_outliers:
                # Filtra outlier
                df_gruppi = df_gruppi[df_gruppi['z_score_scostamento'].abs() <= z_threshold]
                # Ricalcola statistiche e Z-Score su dati puliti
                df_gruppi['z_score_scostamento'] = calculate_z_score_series(df_gruppi['Scostamento VEL_MEDIA %'], robust=use_robust)
            
            df_gruppi['z_score_scostamento'] = df_gruppi['z_score_scostamento'].round(2)
            col_z, col_box = st.columns([3,2])
            with col_z:
                st.plotly_chart(grafico_scostamenti_z_score(df_gruppi), use_container_width=True, key=f"scost_z_{reparto}_{gruppi}")
            with col_box:
                st.plotly_chart(boxplot_scostamenti(df_gruppi), use_container_width=True, key=f"boxplot_scost_{reparto}_{gruppi}")
            st.dataframe(df_gruppi)
            # Inserisci pulsante per scaricare dati in Excel
            excel_data = to_excel_bytes(df_gruppi)
            st.download_button(
                label="📥 Scarica dati gruppo",
                data=excel_data,
                file_name=f'Dati_{reparto}_{gruppi}.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                key=f"download_{reparto}_{gruppi}"
            )
            # inserire calcolo KPI per ogni Risorsa SAS
            st.write("**KPI Scostamento % Velocità Media**")
            
            # Velocità media = Quantità / (Produzione + Attese)
            # Calcolo velocità media consuntiva e prevista per ogni Risorsa SAS
            df_kpi = df_gruppi.groupby('Risorsa SAS').agg({
                'QTA_PRODOTTA': 'sum',
                'PRODUZIONE_CONS': 'sum',
                'ATTESE_CONS': 'sum',
                'QTA_PREVISTA': 'sum',
                'PRODUZIONE_PREV': 'sum',
                'ATTESE_PREV': 'sum'
            }).reset_index()
            df_kpi['VEL_MEDIA_CONS'] = df_kpi['QTA_PRODOTTA'] / (df_kpi['PRODUZIONE_CONS'] + df_kpi['ATTESE_CONS']).replace(0, np.nan)
            df_kpi['VEL_MEDIA_PREV'] = df_kpi['QTA_PREVISTA'] / (df_kpi['PRODUZIONE_PREV'] + df_kpi['ATTESE_PREV']).replace(0, np.nan)
            df_kpi['SCOSTAMENTO_%'] = (df_kpi['VEL_MEDIA_CONS'] - df_kpi['VEL_MEDIA_PREV']) / df_kpi['VEL_MEDIA_PREV'].replace(0, np.nan) * 100
            # Visualizza KPI come metriche in colonne
            cols_kpi = st.columns(len(df_kpi))
            for idx, (index, row) in enumerate(df_kpi.iterrows()):
                with cols_kpi[idx]:
                    st.metric(
                        label=row['Risorsa SAS'], 
                        value=f"{row['VEL_MEDIA_PREV']:.0f}", 
                        delta=f"{row['SCOSTAMENTO_%']:.2f} %", 
                        border=True
                    )
            #st.dataframe(df_kpi)

        st.markdown("---")

    st.subheader('Analisi capoconti', divider='grey')

    lean_filtered.reset_index(drop=True, inplace=True)
    lean_filtered['TEMPO_PREVISTO'] = lean_filtered['PRODUZIONE_PREV'] + lean_filtered['ATTESE_PREV']
    lean_filtered['TEMPO_CONS'] = lean_filtered['PRODUZIONE_CONS'] + lean_filtered['ATTESE_CONS']

    st.write('Numero capiconti: ', lean_filtered['CAPOCONTO'].nunique())

    #lean_filtered

    capoconto_pivot = lean_filtered.pivot_table(index='CAPOCONTO', columns=['COD_REPARTO'], values=['QTA_PREVISTA','TEMPO_PREVISTO','PRODUZIONE_PREV', 'ATTESE_PREV',
                                                'QTA_PRODOTTA', 'TEMPO_CONS','PRODUZIONE_CONS', 'ATTESE_CONS'], aggfunc='sum')

    # Calcolo velocità medie per ogni reparto
    # Estrai i livelli univoci dei reparti (level 1 delle colonne)
    reparti_pivot = capoconto_pivot.columns.get_level_values(1).unique()

    for reparto in reparti_pivot:
        # Calcolo VEL_MEDIA_PREV
        try:
            qta_prev = capoconto_pivot[('QTA_PREVISTA', reparto)]
            tempo_prev = capoconto_pivot[('TEMPO_PREVISTO', reparto)]
            # Gestione divisione per zero
            capoconto_pivot[('VEL_MEDIA_PREV', reparto)] = qta_prev / tempo_prev.replace(0, np.nan)
        except KeyError:
            pass # Se mancano le colonne
        
        # Calcolo VEL_MEDIA_CONS
        try:
            qta_cons = capoconto_pivot[('QTA_PRODOTTA', reparto)]
            tempo_cons = capoconto_pivot[('TEMPO_CONS', reparto)]
            # Gestione divisione per zero
            capoconto_pivot[('VEL_MEDIA_CONS', reparto)] = qta_cons / tempo_cons.replace(0, np.nan)
        except KeyError:
            pass

    # Arrotonda e visualizza
    capoconto_pivot = capoconto_pivot.round(2)
    # st.write('capoconto_pivot')
    # st.dataframe(capoconto_pivot)

    # Crea dataframe scostamenti
    # Crea dataframe scostamenti e df_plot per grafico
    reparti_pivot = capoconto_pivot.columns.get_level_values(1).unique()
    dict_scostamenti = {}
    lista_plot_data = [] # Lista per raccogliere dati per df_plot
    
    for reparto in reparti_pivot:
        try:
            vel_prev = capoconto_pivot[('VEL_MEDIA_PREV', reparto)]
            vel_cons = capoconto_pivot[('VEL_MEDIA_CONS', reparto)]
            qta_prod = capoconto_pivot[('QTA_PRODOTTA', reparto)].fillna(0)
            
            # Calcolo scostamento %
            scostamento = (vel_cons - vel_prev) / vel_prev.replace(0, np.nan)
            dict_scostamenti[reparto] = scostamento
            
            # Prepare data for plot (long format)
            df_temp = pd.DataFrame({
                'CAPOCONTO': capoconto_pivot.index,
                'Reparto': reparto,
                'Scostamento %': scostamento,
                'QTA_PRODOTTA': qta_prod
            })
            lista_plot_data.append(df_temp)
            
        except KeyError:
            continue
            
    capoconto_scostamento = pd.DataFrame(dict_scostamenti, index=capoconto_pivot.index)
    
    # Visualizza dataframe formattato come percentuale
    # st.write("Scostamento % Velocità per Reparto:")
    # st.dataframe(capoconto_scostamento.style.format("{:.2%}"), use_container_width=True)
    
    st.subheader('Analisi Anomalie Scostamento (Strip Plot)', divider='grey')
    
    # Preparazione dati per grafico
    if lista_plot_data:
        df_plot = pd.concat(lista_plot_data, ignore_index=True)
        df_plot = df_plot.dropna(subset=['Scostamento %'])
    else:
        df_plot = pd.DataFrame(columns=['CAPOCONTO', 'Reparto', 'Scostamento %', 'QTA_PRODOTTA'])

    # UI Control for Strip Plot Outliers
    col_strip_toggle, col_strip_slider = st.columns([1, 2])
    with col_strip_toggle:
        remove_strip_outliers = st.toggle("Rimuovi outlier Strip Plot", key="strip_toggle")
    with col_strip_slider:
        if remove_strip_outliers:
            z_strip_threshold = st.slider("Soglia Z-Score", min_value=3.0, max_value=10.0, value=5.0, step=0.1, key="strip_slider")
            
    if remove_strip_outliers:
        # Calcola Z-Score Robusto per ogni reparto su 'Scostamento %'
        df_plot['z_score'] = df_plot.groupby('Reparto')['Scostamento %'].transform(lambda x: calculate_z_score_series(x, robust=True))
        df_plot = df_plot[df_plot['z_score'].abs() <= z_strip_threshold]
    
    # Creazione Strip Plot (usiamo scatter per avere size variabile)
    fig_strip = px.scatter(
        df_plot, 
        x='Reparto', 
        y='Scostamento %', 
        color='Scostamento %',
        color_continuous_scale='RdYlGn',
        size='QTA_PRODOTTA',
        size_max=40,
        hover_name='CAPOCONTO',
        hover_data={'Reparto': True, 'Scostamento %': ':.2%', 'QTA_PRODOTTA': True},
        title='Distribuzione Scostamenti Velocità per Reparto (Bubble Plot)',
        height=600
    )
    
    # Aggiungi linee di riferimento per soglie di attenzione
    fig_strip.add_hline(y=0.20, line_dash="dash", line_color="orange", annotation_text="+20%")
    fig_strip.add_hline(y=-0.20, line_dash="dash", line_color="orange", annotation_text="-20%")
    fig_strip.add_hline(y=0.50, line_dash="dot", line_color="red", annotation_text="+50%")
    fig_strip.add_hline(y=-0.50, line_dash="dot", line_color="red", annotation_text="-50%")

    # Formattazione asse Y in percentuale e legenda
    fig_strip.update_layout(
        yaxis_tickformat='.0%',
        coloraxis_colorbar=dict(tickformat='.0%')
    )
    fig_strip.update_xaxes(tickfont=dict(size=16))
    
    st.plotly_chart(fig_strip, use_container_width=True)

    capoconto_pivot_velocità = capoconto_pivot[['VEL_MEDIA_PREV', 'VEL_MEDIA_CONS']].fillna(0).astype(int)
    
    with st.expander("Dati velocità media per capoconto"):
        # visualizza dataframe come intero senza decimali
        st.dataframe(capoconto_pivot_velocità)

    st.subheader('Andamento velocità media per capoconto')
    # selezione capoconto
    capoconto_selezionato = st.selectbox('Seleziona capoconto', capoconto_pivot_velocità.index)
    
    # crea grafico andamento velocità media per capoconto
    if capoconto_selezionato:
        # Filtra lean per il capoconto selezionato
        df_cap = lean[lean['CAPOCONTO'] == capoconto_selezionato].copy()
        
        # Calcolo Velocità Media Consuntiva
        # VEL_MEDIA_CONS = QTA_PRODOTTA / (PRODUZIONE_CONS + ATTESE_CONS)
        df_cap['TEMPO_TOT'] = df_cap['PRODUZIONE_CONS'] + df_cap['ATTESE_CONS']
        # Gestione divisione per zero
        df_cap['VEL_MEDIA_CONS'] = df_cap['QTA_PRODOTTA'] / df_cap['TEMPO_TOT'].replace(0, np.nan)

        # Calcolo Velocità Media Prevista
        # VEL_MEDIA_PREV = QTA_PREVISTA / (PRODUZIONE_PREV + ATTESE_PREV)
        df_cap['TEMPO_TOT_PREV'] = df_cap['PRODUZIONE_PREV'] + df_cap['ATTESE_PREV']
        df_cap['VEL_MEDIA_PREV'] = df_cap['QTA_PREVISTA'] / df_cap['TEMPO_TOT_PREV'].replace(0, np.nan)
        
        # Data sorting
        df_cap.sort_values(by='DATA_CHIUSURA', inplace=True)
        
        # Identifica reparti univoci
        reparti_coinvolti = df_cap['COD_REPARTO'].unique()
        
        for reparto in reparti_coinvolti:
            df_rep = df_cap[df_cap['COD_REPARTO'] == reparto]
            
            # Rimuovi NaN per il grafico (se entrambi sono NaN)
            df_rep = df_rep.dropna(subset=['VEL_MEDIA_CONS', 'VEL_MEDIA_PREV'], how='all')
            
            if not df_rep.empty:
                fig_trend = px.line(
                    df_rep, 
                    x='DATA_CHIUSURA', 
                    y=['VEL_MEDIA_CONS', 'VEL_MEDIA_PREV'], 
                    title=f'Andamento Velocità Media - Reparto: {reparto}',
                    markers=True,
                    labels={'value': 'Velocità Media (E/(A+C))', 'DATA_CHIUSURA': 'Data Chiusura', 'variable': 'Tipo Velocità'},
                    height=400
                )
                st.plotly_chart(fig_trend, use_container_width=True)
    

    # st.dataframe(lean_filtered)
    
    

    #=====================================================================================================================================



with tab1:
    # visualizza budget
    # st.write('Budget 2026:')
    # st.dataframe(budget)
    # raggruppa per Gruppo_risorse e calcola la somma delle colonne numeriche (i mesi del budget)
    colonne_numeriche = budget.select_dtypes(include='number').columns.tolist()
    budget_grouped = budget.groupby('Gruppo_risorse')[colonne_numeriche].sum().reset_index()
    #budget_stampa = budget_grouped[budget_grouped['Gruppo_risorse'] == 'STAMPA'].reset_index(drop=True)
    # st.dataframe(budget_grouped)
    # filtra per Gruppo_risorse = STAMPA
    budget_stampa = budget_grouped[budget_grouped['Gruppo_risorse'] == 'STAM'].reset_index(drop=True)
    #st.dataframe(budget_stampa)
    
    # Converti colonna Month in data
    # Formato atteso Jan.24
    try:
        db_raw['Month'] = pd.to_datetime(db_raw['Month'], format='%b.%y')
    except Exception as e:
        st.error(f"Errore conversione data: {e}")


    db_raw.rename(columns={'Month': 'Periodo'}, inplace=True)

    # converti in formato 2025-01
    try:
        db_raw['Periodo'] = db_raw['Periodo'].dt.strftime('%Y-%m')
    except Exception as e:
        st.error(f"Errore conversione data: {e}")



    # Pulizia colonna Reparto (rimuovi duplicazioni tipo "ACCO ACCO")
    def clean_reparto_string(val):
        if not isinstance(val, str):
            return val
        words = val.split()
        n = len(words)
        if n > 0 and n % 2 == 0:
            half = n // 2
            # Controlla se la prima metà è uguale alla seconda
            if words[:half] == words[half:]:
                return " ".join(words[:half])
        return val

    if 'Reparto' in db_raw.columns:
        db_raw['Reparto'] = db_raw['Reparto'].apply(clean_reparto_string)


    # Elimina le colonne che iniziano con "G."
    for col in db_raw.columns:
        if col.startswith('G.'):
            db_raw.drop(col, axis=1, inplace=True)


    macchine_obsolete = ['A150 KMN FINEST. 105/2','A169 VEGA 80_2','A168 VEGA 80_1','A160 CTPACK 1','A163 BST ALPINA 75 CTPK_3']
    db_raw = db_raw[~db_raw['Risorsa SAS'].isin(macchine_obsolete)]

    # merge abbinamento con db_raw
    db_raw = pd.merge(db_raw, abbinamento[['Risorsa SAS', 'Gruppo', 'Plant','VLL_budget']], on='Risorsa SAS', how='left')

    db_raw['KPI_VLL'] = (((db_raw['Velocità Lordissima (E/(A+B+C+H)'] / db_raw['VLL_budget'])-1)*100).round(2)

    st.write('Anteprima dati caricati:')
    st.write('Numero di righe totali:', len(db_raw))
    st.dataframe(db_raw)

    # Crea pulsante per scaricare dati
    OEE_excel = to_excel_bytes(db_raw)
    st.download_button(
        label="📥 Scarica dati OEE",
        data=OEE_excel,
        file_name='Dati_OEE.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    st.markdown("---")
    st.markdown("### Andamento FF stampati 2024 - YTD | YoY | Budget 2026")
    with st.expander("Andamento mensile e YOY"):
        df_FF_mese = db_raw[db_raw['Reparto'] == 'STAM']
        df_FF_mese = df_FF_mese.groupby('Periodo')['Volume Prod. (E)'].sum().reset_index()
        # Calcolo etichetta in milioni con 1 decimale
        df_FF_mese['Label'] = (df_FF_mese['Volume Prod. (E)'] / 1000000).map('{:,.1f} M'.format)

        fig_FF_mese = px.bar(df_FF_mese, x='Periodo', y='Volume Prod. (E)', text='Label', title='Andamento FF stampati',
        height=600)
        # aggiungi etichetta valori sopra le barre espresso in milioni
        fig_FF_mese.update_traces(textposition='outside', textfont_size=16)
        st.plotly_chart(fig_FF_mese, use_container_width=True)


        df_FF_yoy = df_FF_mese.copy()
        df_FF_yoy['Year'] = df_FF_yoy['Periodo'].apply(lambda x: x.split('-')[0])
        df_FF_yoy['Month'] = df_FF_yoy['Periodo'].apply(lambda x: x.split('-')[1])

        fig_yoy = px.line(df_FF_yoy, x='Month', y='Volume Prod. (E)', color='Year', markers=True, 
                        title='Confronto Fogli stampati YOY', text='Label',
                        category_orders={"Month": ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]},
                        height=600)
        fig_yoy.update_traces(textposition="top center", marker_size=12, textfont_size=14)

        # Aggiungi linea Budget 2026
        # Reshape budget_stampa da wide a long: le colonne numeriche = mesi in ordine gen→dic
        colonne_mesi_budget = [col for col in budget_stampa.columns if col != 'Gruppo_risorse']
        mesi_label = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
        budget_long = budget_stampa.melt(id_vars='Gruppo_risorse', value_vars=colonne_mesi_budget,
                                         var_name='col_mese', value_name='Volume Budget')
        # Assegna etichetta mese in base alla posizione della colonna
        budget_long['Month'] = budget_long.groupby('Gruppo_risorse').cumcount().map(
            lambda i: mesi_label[i] if i < len(mesi_label) else None
        )
        budget_long = budget_long.dropna(subset=['Month'])

        fig_yoy.add_trace(go.Scatter(
            x=budget_long['Month'],
            y=budget_long['Volume Budget'],
            mode='lines+markers',
            name='Budget 2026',
            line=dict(color='gray', dash='dash', width=2),
            marker=dict(size=8, color='gray'),
        ))

        st.plotly_chart(fig_yoy, use_container_width=True)
    st.markdown("---")


    # Inserisci toggle per scegliere plant
    plant_selezionato = st.radio('Plant', db_raw['Plant'].unique(), horizontal=True)

    # Analisi Cavriago =================
    st.header(f'Analisi {plant_selezionato} per Gruppo Risorse', divider='gray')

    df_plant = db_raw[db_raw['Plant'] == plant_selezionato]


    sequenza =  ['STAM','ACCO',  'FUST', 'INCO','OROC','FINE']

    dizionario_reparti = {'STAM': 'STAMPA', 'FUST': 'FUSTELLATURA', 'INCO': 'PIEGA-INCOLLA', 'OROC': 'ORO A CALDO', 'FINE': 'FINESTRATURA', 'ACCO': 'ONDULATORE'}

    reparti = sorted(list(df_plant['Reparto'].unique()), key=lambda x: sequenza.index(x) if x in sequenza else len(sequenza))

    # Funzione per grafico analisi Gruppo
    def grafico_analisi_gruppo(df, gruppo):
        df_gruppo = df[df['Gruppo'] == gruppo]
        
        # Prepara dati per marker (dimensione e testo) per la serie Tiratura
        if not df_gruppo.empty:
            max_vol = df_gruppo['Volume Prod. (E)'].max()
            if max_vol == 0:
                max_vol = 1
            # Scala dimensione: min 8, max 35
            marker_sizes = 8 + (df_gruppo['Volume Prod. (E)'] / max_vol * 27)
            # Testo formattato: "1.23 M"
            marker_text = (df_gruppo['Volume Prod. (E)'] / 1000000).apply(lambda x: f'{x:.2f} M')
        else:
            marker_sizes = 10
            marker_text = ""

        # Crea grafico con doppio asse Y
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Trace 1: Velocità Lordissima (Primary Y) - Marker costanti, no testo
        fig.add_trace(
            go.Scatter(
                x=df_gruppo['Periodo'],
                y=df_gruppo['Velocità Lordissima (E/(A+B+C+H))'],
                name='Velocità Lordissima',
                mode='lines+markers',
                marker=dict(
                    size=10,
                    color='blue',
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                line=dict(color='blue')
            ),
            secondary_y=False
        )
        
        # Trace 2: Tiratura Media (Secondary Y) - Marker variabili, testo, colore su Avviamenti
        fig.add_trace(
            go.Scatter(
                x=df_gruppo['Periodo'],
                y=df_gruppo['Tiratura Media (E/G)'],
                name='Tiratura Media',
                mode='lines+markers+text',
                text=marker_text,
                textposition="top center",
                marker=dict(
                    size=marker_sizes,
                    color=df_gruppo['Numero Avviamenti Primari (G)'],
                    colorscale='RdYlGn_r', # Green=Low, Red=High
                    showscale=True,
                    colorbar=dict(title="N. Avviamenti (G)", x=1.15),
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                line=dict(color='red', dash='dot')
            ),
            secondary_y=True
        )
        
        # Update Layout
        fig.update_layout(
            title=f'Analisi Gruppo: {gruppo}',
            width=900,
            height=750,
            margin=dict(b=100),
            legend=dict(y=1.15, x=1.1),
            annotations=[
                dict(
                    text="Dimensione del marker: volume",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=1.1,
                    y=-0.1,
                    xanchor='right',
                    yanchor='top',
                    font=dict(size=16)
                )
            ]
        )
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Velocità Lordissima", secondary_y=False)
        fig.update_yaxes(title_text="Tiratura Media", secondary_y=True)

        st.plotly_chart(fig, use_container_width=True)
        # crea grafico andamento
        
    def grafico_andamento_risorsa(df, gruppo):
        # line chart per gruppo
        fig_andamento = px.line(df, x='Periodo', y='Velocità Lordissima (E/(A+B+C+H))', color='Risorsa SAS', markers=True, 
                    title=f'Andamento Velocità Lordissima (E/(A+B+C+H)) per {gruppo}')
        st.plotly_chart(fig_andamento, use_container_width=True)
    #



    for reparto in reparti:
        df_plant_REPARTO = df_plant[df_plant['Reparto'] == reparto].groupby(['Gruppo','Periodo']).agg({
        'Volume Prod. (E)': 'sum',
        'Ore Totali (A+B+C+H)': 'sum',
        'Numero Avviamenti Primari (G)': 'sum',
        }).reset_index()
        df_plant_REPARTO['Velocità Lordissima (E/(A+B+C+H))'] = df_plant_REPARTO['Volume Prod. (E)'] / df_plant_REPARTO['Ore Totali (A+B+C+H)']
        df_plant_REPARTO['Tiratura Media (E/G)'] = df_plant_REPARTO['Volume Prod. (E)'] / df_plant_REPARTO['Numero Avviamenti Primari (G)']
        st.markdown(f'#### Andamento {dizionario_reparti[reparto]}')
        
        
        lista_gruppi = df_plant_REPARTO['Gruppo'].unique()
        for gruppo in lista_gruppi:
            grafico_analisi_gruppo(df_plant_REPARTO, gruppo)

        with st.expander(f"Dati {dizionario_reparti[reparto]} e grafico tiratura media vs velocità lordissima"):
            col_a, col_b = st.columns([3,2])
            with col_a:
                st.dataframe(df_plant_REPARTO)
                # calcola la mediana di Velocità lordissima per anno
                df_plant_REPARTO['Anno'] = pd.to_datetime(df_plant_REPARTO['Periodo']).dt.year
                lista_mediane = df_plant_REPARTO.groupby('Anno')['Velocità Lordissima (E/(A+B+C+H))'].median().reset_index()
                lista_mediane['Anno'] = lista_mediane['Anno'].astype(int)
                lista_mediane = lista_mediane.sort_values(by='Anno')
                st.dataframe(lista_mediane)
            with col_b:
                # crea scatter plot tiratura media vs velocità lordissima
                # Rimuovi infiniti e NaN prima del grafico per evitare errori in trendline
                df_scatter = df_plant_REPARTO.replace([np.inf, -np.inf], np.nan).dropna(subset=['Tiratura Media (E/G)', 'Velocità Lordissima (E/(A+B+C+H))'])
                fig_scatter = px.scatter(df_scatter, x='Tiratura Media (E/G)', y='Velocità Lordissima (E/(A+B+C+H))', color='Gruppo', size='Volume Prod. (E)', hover_data=['Gruppo', 'Periodo'], trendline="ols")
                
                # Aggiungi R2 alla legenda
                try:
                    results = px.get_trendline_results(fig_scatter)
                    for i, row in results.iterrows():
                        fit = row['px_fit_results']
                        if fit:
                            r2 = fit.rsquared
                            group = row['Gruppo']
                            for trace in fig_scatter.data:
                                if trace.name == group:
                                    trace.name = f"{group} (R²={r2:.1%})"
                except Exception as e:
                    st.warning(f"Errore calcolo trendline: {e}")

                fig_scatter.update_layout(height=700)
                st.plotly_chart(fig_scatter, use_container_width=True)



    # Seleziona Reparto
    st.subheader('Analisi per Reparto', divider='gray')
    reparti_da_scegliere = df_plant['Reparto'].unique()

    # Get available department codes
    codes_present = df_plant['Reparto'].unique()

    # Sort codes based on defined sequence
    sorted_codes = sorted(codes_present, key=lambda x: sequenza.index(x) if x in sequenza else len(sequenza))

    # Map sorted codes to names for display
    reparti_names = [dizionario_reparti.get(code, code) for code in sorted_codes]

    # Radio button selects from names
    reparto_nome_selezionato = st.radio('Seleziona Reparto', reparti_names, horizontal=True)

    # Find code corresponding to selected name
    reparto_selezionato = sorted_codes[reparti_names.index(reparto_nome_selezionato)]

    # Filtra dati



    df_reparto = df_plant[df_plant['Reparto'] == reparto_selezionato]

    # Andamento OEE
    st.subheader(f'Andamento OEE e velocità per Reparto - {reparto_selezionato}')

    for gruppo in df_reparto['Gruppo'].unique():
        df_reparto_gruppo = df_reparto[df_reparto['Gruppo'] == gruppo]
        fig_oee = px.line(df_reparto_gruppo, x='Periodo', y='OEE', color='Risorsa SAS', markers=True, 
                        title=f'Andamento OEE per {gruppo}',
                        height=700)
        st.plotly_chart(fig_oee, use_container_width=True)

    # Seleziona il tipo di velocità
    col_velocità = [col for col in df_reparto.columns if 'Velocità' in col]
    velo_selezionata = st.radio('Seleziona tipo velocità', col_velocità, horizontal=True, index=2)

    col_c, col_d = st.columns([3,2])
    with col_c:
        for gruppo in df_reparto['Gruppo'].unique():
            df_reparto_gruppo = df_reparto[df_reparto['Gruppo'] == gruppo]
            fig_velocità = px.line(df_reparto_gruppo, x='Periodo', y=velo_selezionata, color='Risorsa SAS', markers=True, 
                        title=f'Andamento {velo_selezionata} per {gruppo}',
                        height=700)
            st.plotly_chart(fig_velocità, use_container_width=True)
    with col_d:
        # boxplot velocità per Risorsa SAS
        for gruppo in df_reparto['Gruppo'].unique():
            df_reparto_gruppo = df_reparto[df_reparto['Gruppo'] == gruppo]
            fig_boxplot = px.box(df_reparto_gruppo, x='Risorsa SAS', y=velo_selezionata, color='Risorsa SAS', 
                        title=f'Boxplot {velo_selezionata} per {gruppo}',
                        height=700)
            st.plotly_chart(fig_boxplot, use_container_width=True)

    # Calcola il mese precedente rispetto all'ultima data disponibile nel dataframe
    periodo_max = df_reparto['Periodo'].max()  # formato 'YYYY-MM'
    data_max = pd.to_datetime(periodo_max, format='%Y-%m')
    mese_precedente = data_max - pd.DateOffset(months=1)
    ultimo_periodo = mese_precedente.strftime('%Y-%m')
    
    df_reparto_ultimo_periodo = df_reparto[df_reparto['Periodo'] == ultimo_periodo]

    st.subheader(f'KPI {ultimo_periodo} - {dizionario_reparti[reparto_selezionato]} Velocità lordissima vs Budget')


    row = st.container()
    with row:
        risorse = df_reparto_ultimo_periodo['Risorsa SAS'].unique()
        cols = st.columns(len(risorse))
        for index, risorsa in enumerate(risorse):
            df_risorsa = df_reparto_ultimo_periodo[df_reparto_ultimo_periodo['Risorsa SAS'] == risorsa]
            valore = df_risorsa['Velocità Lordissima (E/(A+B+C+H)'].item()
            delta_val = df_risorsa['KPI_VLL'].item()
            
            cols[index].metric(
                label=risorsa, 
                value=f"{valore:.0f}", 
                delta=f"{delta_val:.2f} % vs budget", 
                border=True
            )

    st.stop()

