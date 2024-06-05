import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from neuralprophet import NeuralProphet
import pickle

class WebuiHandler:
    def __init__(self):
        self.model = NeuralProphet()
        self.setup_interface()
        
    def setup_interface(self):
        self.inputs = [
            gr.components.File(label='Upload CSV'),
            gr.components.Number(value=30, label='Prediction Periods'),
        ]

        self.outputs = [
            gr.components.Plot(label='CSV Graph')
        ]
        
        self.interface = gr.Interface(fn=self.handle_inputs, 
                                      inputs=self.inputs, 
                                      outputs=self.outputs)
        
    def handle_inputs(self, file, periods):
        data = self.read_csv(file)
        self.train_model(data)
        forecast = self.predict(data, periods)
        fig = self.plot_graph(data, forecast)
        return fig
    
    def read_csv(self, file):
        data = pd.read_csv(file.name)
        return data
    
    def train_model(self, data):
        # Fit the NeuralProphet model
        # Assuming the CSV has 'ds' (date) and 'y' (value) columns
        self.model.fit(data, freq='D')
    
    def predict(self, data, periods=30):
        # Make future dataframe
        future = self.model.make_future_dataframe(data, periods=periods)
        # Make predictions
        forecast = self.model.predict(future)
        return forecast
    
    def plot_graph(self, data, forecast):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat1'], mode='lines', name='Forecast'))
        
        fig.update_layout(title='Actual Data and Forecast',
                          xaxis_title='Date',
                          yaxis_title='Value')
        return fig
        
    def launch(self):
        self.interface.launch(share=False)