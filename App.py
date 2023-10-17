import streamlit as st
import plotly.express as px
from fastcore.all import *
import pandas as pd
import numpy as np
from scipy import optimize

pd.set_option('float_format', '{:.2f}'.format)

# %% ../nbs/03_Mortgages.ipynb 5


class Mortgage():
    """
    Class tho represents teh mortgage values.
    """

    def __init__(self,
                 taxa: float,
                 years: float,
                 total: int,
                 entry: int,
                 MIP: float = 0,
                 DFI: float = 0,
                 TCA: int = 25):

        self.taxa = taxa
        self.monthly_t = self.taxa/12/100
        self.years = years
        self.total = total
        self.entry = entry
        self.debt = self.total - self.entry
        self.months = round(self.years*12)
        self.TCA = TCA
        self.MIP = MIP
        self.DFI = DFI

# %% ../nbs/03_Mortgages.ipynb 6


class Mortgage_sac(Mortgage):
    """
Class representing hte MOrtgae with constant amortizatio system
    """

    def __init__(self,
                 taxa: float,
                 years: float,
                 total: int = 0,
                 entry: int = 0,
                 MIP: float = 0,
                 DFI: float = 0,
                 TCA: int = 25):
        super().__init__(
            taxa,
            years,
            total,
            entry,
            MIP,
            DFI,
            TCA)
        self.devedor = self.total - self.entry


# %% ../nbs/03_Mortgages.ipynb 7
@patch
def calculate_amortization(
    self: Mortgage_sac,
) -> float:
    """
    Function to calculate the amortization value 
    """
    return self.debt/self.months


# %% ../nbs/03_Mortgages.ipynb 8
@patch
def calculate_mortage(
    self: Mortgage_sac,
) -> tuple:
    mortgage = np.zeros((self.months, 8))
    mortgage[0, 0] = self.devedor
    mortgage[:, 2] = self.calculate_amortization()  # amort
    mortgage[:, 1] = np.cumsum(mortgage[:, 2])  # amort cumulativa
    mortgage[1:, 0] = self.devedor - mortgage[:-1, 1]  # devedor
    mortgage[:, 3] = mortgage[:, 0]*(self.monthly_t)  # juros
    mortgage[:, 4] = self.MIP  # mortgage[:,2] *
    mortgage[:, 5] = self.DFI  # mortgage[:,2] *
    mortgage[:, 6] = self.TCA
    # + mortgage[:,3] + mortgage[:,4] + mortgage[:,5] + self.TCA
    mortgage[:, 7] = mortgage[:, 2:].sum(1)
    self.mortgage = pd.DataFrame(mortgage, columns=[
                                 'devedor',  'amort_cum', 'amortizacion',  'juros', 'MIP', 'DFI', 'TCA', 'total'])
    mid_i = self.mortgage.shape[0]//2
    return (mortgage[:7].sum(), mortgage[0, 7], mortgage[mid_i, 7], mortgage[-1, 7])


# %% ../nbs/03_Mortgages.ipynb 10
class Mortgage_price(Mortgage):

    def __init__(self,
                 taxa: float,
                 years: float,
                 total: int = 0,
                 entry: int = 0,
                 MIP: float = 0,
                 DFI: float = 0,
                 TCA: int = 25):
        super().__init__(
            taxa,
            years,
            total,
            entry,
            MIP,
            DFI,
            TCA)
        self.devedor = self.total - self.entry
        self.parcela = self.calculate_parcela()

# %% ../nbs/03_Mortgages.ipynb 11


@patch
def calculate_parcela(self: Mortgage_price):
    return self.debt / (((1+self.monthly_t)**self.months - 1) / ((1+self.monthly_t)**self.months * self.monthly_t))


# %% ../nbs/03_Mortgages.ipynb 12
@patch
def calculate_mortage(
    self: Mortgage_price
) -> tuple:  # Return tuple with , accumulated payment, total amount, first payment,

    mortgage = np.zeros((self.months, 8))
    # mortgage[:,1] = self.calculate_parcela()
    mortgage[:, 4] = self.MIP
    mortgage[:, 5] = self.DFI
    mortgage[:, 6] = self.TCA
    for i in range(0, self.months):
        mortgage[i, :] = self.update_values(mortgage[i, :])
    # mortgage[:,7] = np.cumsum(mortgage[:,3])
    mortgage[:, 1] = mortgage[:, 2].cumsum()
    self.mortgage = pd.DataFrame(mortgage,
                                 columns=['devedor',  'amort_cum', 'amortizacion',  'juros', 'MIP', 'DFI', 'TCA', 'total'])
    mid_i = self.mortgage.shape[0]//2
    return (mortgage[:7].sum(), mortgage[0, 7], mortgage[mid_i, 7], mortgage[-1, 7])

# %% ../nbs/03_Mortgages.ipynb 13


@patch
def update_values(self: Mortgage_price, values: np.ndarray):
    values[0] = self.devedor  # current debt
    values[3] = values[0]*self.monthly_t  # current juros
    values[2] = self.parcela - values[3]  # current amortizacion
    values[7] = values[2:].sum()  # total
    self.devedor = self.devedor - values[3]

    return (values)


# %% ../nbs/03_Mortgages.ipynb 14
@patch
def calcular_debt(self: Mortgage_price, parcela: float):
    return round((parcela - self.MIP - self.DFI - self.TCA) * (((1+self.monthly_t)**self.months - 1) / ((1+self.monthly_t)**self.months * self.monthly_t)), 2)


def calcular_prestamo(m_type, taza, anos, value, entrada):
    if m_type == 'SAC':
        mortgage = Mortgage_sac(taza, anos, value, entrada, 40.32, 26)
    else:
        mortgage = Mortgage_price(taza, anos, value, entrada, 40.32, 26)
    total, first, mid, last = mortgage.calculate_mortage()
    print(total)
    return mortgage.mortgage, total.round(2)


def calcular_cuota(taza, anos, cuota, total):
    mortgage = Mortgage_price(taza, anos, MIP=40.32, DFI=26, TCA=25)
    value = mortgage.calcular_debt(cuota)
    increment = abs(total - value)
    total = (value + increment)
    entrada = total - value
    mortgage = Mortgage_price(
        taza, anos, total, entrada, 40.32, 26)
    mortgage.calculate_mortage()
    return mortgage.mortgage, total, entrada


mortgage_type = st.sidebar.selectbox('Kind', ['SAC', 'PRICE'])
value = st.sidebar.number_input('Valor propiedad', value=200000)
entrada = st.sidebar.number_input('Entrada', value=40000)
anos = st.sidebar.number_input('Años', value=35)
taza = st.sidebar.number_input('taxa', value=10.06)


calculateButton = st.sidebar.button('Calcular Prestamo', 'Calcular_cuotas')

cuota = st.sidebar.number_input('Valor Maximo', value=1500)

calcularPrestamo = st.sidebar.button('Calcular prestamos', 'Calcular_prestamo')


if calculateButton:
    df, total = calcular_prestamo(mortgage_type, taza, anos, value, entrada)
    a, b = st.columns(2)
    with a:
        st.metric('Valor Total Pagado', f"BRL {total}")
    st.dataframe(df.round(2))
    plot = px.line(
        data_frame=df,
        x=df.index,
        y=['amortizacion', 'juros'],
        labels={
            'amortizacion': 'Amortización',
            'juros': 'Intereses',
            'devedor': 'Saldo Devedor',
            'index': 'Meses',
            'value': 'Reales'})
    st.plotly_chart(plot)

elif calcularPrestamo:
    # print(calcular_cuota(taza, anos, cuota))
    df, total, entrada = calcular_cuota(taza, anos, cuota, value)
    a, b, c = st.columns(3)
    with a:
        st.metric('Valor Cuota', round(cuota, 2))
    with b:
        st.metric('Valor Propiedad', round(total, 2))
    with c:
        st.metric('Valor Entrada', round(entrada, 2))

    st.dataframe(df)
    plot = px.line(
        data_frame=df,
        x=df.index,
        y=['amortizacion', 'juros'],
        labels={
            'amortizacion': 'Amortización',
            'juros': 'Intereses',
            'devedor': 'Saldo Devedor',
            'index': 'Meses',
            'value': 'Reales'})
    st.plotly_chart(plot)
