import numpy as np
import plotly.graph_objects as go
import ETL

class Plot_data:
    def __init__(self, etl:ETL.ETL) -> None:
        self.etl = etl
        self.range_selector = dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="This year", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )

    def plot_fft(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.etl.df['Date'], y=self.etl.df['Close'], mode='lines', name='Close', visible=True))

        for comp in [3, 6, 9, 25, 100]:
            if comp!=9: fig.add_trace(go.Scatter(x=self.etl.fft_df['Date'], y=self.etl.fft_df['ifft_'+str(comp)].apply(np.real), fill=None, mode='lines', name='Fourier transform with {} components'.format(comp), visible='legendonly'))
            else: fig.add_trace(go.Scatter(x=self.etl.fft_df['Date'], y=self.etl.fft_df['ifft_'+str(comp)].apply(np.real), fill=None, mode='lines', name='Fourier transform with {} components'.format(comp), visible=True))
            
        # Personalizar el diseño del gráfico
        fig.update_layout(
            title='SANTANDER Stock Prices FFT',
            xaxis_title='Date',
            yaxis_title='EUR',
            hovermode='x',  # Habilitar interacción en el eje X al pasar el cursor
            showlegend=True,  # Mostrar leyendas de las series
            height=600, 
            width=1200,
            legend=dict(
                x=0.3,  # Ajusta la posición horizontal de la leyenda
                y=1.03,  # Ajusta la posición vertical de la leyenda
                orientation='h'  # Cambia la orientación a horizontal
            )
        )
        fig.update_xaxes(rangeslider_visible=True, rangeselector= self.range_selector)
        fig.show()

    def plot_tech(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.etl.df['Date'], y=self.etl.df['Close'], mode='lines', name='Close', visible=True))

        # Agregar la Media Móvil Simple (SMA)
        fig.add_trace(go.Scatter(x=self.etl.df['Date'], y=self.etl.tech_df['SMA_7'], mode='lines', name='SMA_7', visible='legendonly'))
        fig.add_trace(go.Scatter(x=self.etl.df['Date'], y=self.etl.tech_df['SMA_21'], mode='lines', name='SMA_21', visible='legendonly'))

        # Agregar la Media Móvil Exponencial (EMA)
        fig.add_trace(go.Scatter(x=self.etl.df['Date'], y=self.etl.tech_df['EMA'], mode='lines', name='EMA', visible='legendonly'))

        # Agregar Media Móvil de Convergencia/Divergencia (MACD)
        fig.add_trace(go.Scatter(x=self.etl.df['Date'], y=self.etl.tech_df['MACD'], mode='lines', name='MACD', visible='legendonly'))

        # Agregar las bandas de Bollinger como áreas sombreadas
        fig.add_trace(go.Scatter(x=self.etl.df['Date'], y=self.etl.tech_df['uBB'], fill=None, mode='lines', line_color='rgba(0, 0, 255, 0.1)', name='Bollinger Upper', visible='legendonly'))
        fig.add_trace(go.Scatter(x=self.etl.df['Date'], y=self.etl.tech_df['lBB'], fill='tonexty', mode='lines', line_color='rgba(255, 0, 0, 0.1)', name='Bollinger Lower', visible='legendonly'))

        # Personalizar el diseño del gráfico
        fig.update_layout(
            title='SANTANDER Stock Prices Technical Indicators',
            xaxis_title='Date',
            yaxis_title='EUR',
            hovermode='x',  # Habilitar interacción en el eje X al pasar el cursor
            showlegend=True,  # Mostrar leyendas de las series
            height=600, 
            width=1200,
            legend=dict(
                x=0.3,  # Ajusta la posición horizontal de la leyenda
                y=1.02,  # Ajusta la posición vertical de la leyenda
                orientation='h'  # Cambia la orientación a horizontal
            )
        )
        fig.update_xaxes(rangeslider_visible=True, rangeselector=self.range_selector)
        
        # Mostrar el gráfico
        fig.show()