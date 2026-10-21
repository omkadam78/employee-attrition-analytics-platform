from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title='HR Attrition Analytics', layout='wide')
st.title('Employee Attrition Analytics Platform')
data = pd.read_csv(ROOT / 'data' / 'employee_attrition.csv')
scores = pd.read_csv(ROOT / 'reports' / 'results' / 'scored_test_employees.csv')
department = st.selectbox('Department', ['All'] + sorted(data['Department'].unique().tolist()))
view = data if department == 'All' else data[data['Department'] == department]
c1, c2, c3 = st.columns(3)
c1.metric('Employees', len(view)); c2.metric('Observed attrition rate', f"{(view['Attrition'] == 'Yes').mean():.1%}"); c3.metric('Retention rate', f"{(view['Attrition'] == 'No').mean():.1%}")
st.subheader('Observed attrition by department'); st.bar_chart(data.assign(Left=data['Attrition'].eq('Yes')).groupby('Department')['Left'].mean())
st.subheader('Predicted risk categories on the held-out test set'); st.bar_chart(scores['RiskCategory'].value_counts())
st.subheader('Workforce demographics'); st.dataframe(view.groupby(['Department', 'Gender']).agg(Employees=('Attrition','size'), AverageAge=('Age','mean'), AverageIncome=('MonthlyIncome','mean')).reset_index(), use_container_width=True)
st.subheader('Model performance'); st.dataframe(pd.read_csv(ROOT / 'reports' / 'results' / 'advanced_model_metrics.csv'), use_container_width=True)
st.caption('Observed outcomes and model predictions appear in separate sections. Predictions support review and do not determine employment actions.')
