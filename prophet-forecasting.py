# Databricks notebook source
import pandas as pd
import numpy as np
from fbprophet import Prophet

# COMMAND ----------

df = pd.read_csv('/dbfs/FileStore/lorenzo.baldacci@databricks.com/example_wp_peyton_manning.csv')
df['y'] = np.log(df['y'])
df.head()

# COMMAND ----------

m = Prophet()
m.fit(df);

# COMMAND ----------

future = m.make_future_dataframe(periods=365)
future.tail()

# COMMAND ----------

forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# COMMAND ----------

m.plot(forecast);

# COMMAND ----------

m.plot_components(forecast)