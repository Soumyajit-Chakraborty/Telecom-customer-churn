import streamlit as st
import pandas as pd
import joblib as jb

@st.cache_resource
def load_encoder():
    enc = jb.load('encoder.pkl')
    return enc


@st.cache_resource
def load_model():
    mod = jb.load('churn_predictor.pkl')
    return mod


with st.spinner("Loading, please wait"):
    encoder = load_encoder()
    model = load_model()

with st.sidebar:
    st.header('Enter the details of the customer:')
    int_ser = st.pills('Internet Service', [
                       'Fiber optic', 'DSL', 'No'], key='int_ser')

    if int_ser == 'No':
        online_sec = st.pills('Online Security', [
                              'No internet service'], key='online_sec_no')
        tech_supp = st.pills(
            'Tech Support', ['No internet service'], key='tech_supp_no')
    else:
        online_sec = st.pills('Online Security', [
                          'Yes', 'No', 'No internet service'], key='online_sec')
        tech_supp = st.pills(
        'Tech Support', ['Yes', 'No', 'No internet service'], key='tech_supp')

    cont = st.pills('Contract', ['Month-to-month',
                    'Two year', 'One year'], key='cont')
    pay_meth = st.selectbox('Payment method', [
                            'Electronic check', 'Mailed check', 'Credit card (automatic)', 'Bank transfer (automatic)'], key='pay_meth')
    tenure = st.slider('Tenure', 0, 72, 10, 1, key='tenure')
    m_charge = st.number_input('Montly Charges', key='m_charge')
    done = st.button('Check', type="primary", key='done')

st.title('Telecom Customer Churn Prediction')
st.markdown('#### Start by selecting parameters on the sidebar and hit \'check\'')

if done:

    data = {
        'tenure': [tenure],
        'MonthlyCharges': [m_charge],
        'InternetService': [int_ser],
        'OnlineSecurity': [online_sec],
        'TechSupport': [tech_supp],
        'Contract': [cont],
        'PaymentMethod': [pay_meth]
    }

    df = pd.DataFrame(data)
    cat_columns = ['InternetService', 'OnlineSecurity',
                   'TechSupport', 'Contract', 'PaymentMethod']

    encoded_df = pd.DataFrame(encoder.transform(
        df[['InternetService', 'OnlineSecurity', 'TechSupport', 'Contract', 'PaymentMethod']]), columns=encoder.get_feature_names_out(cat_columns))

    final_df = pd.concat(
        [df[['tenure', 'MonthlyCharges']], encoded_df], axis=1)

    to_show = pd.DataFrame({
        "Fields": ["Tenure", "Monthly Charges", "Internet Service", "Online Security", "Tech Support", "Contract", "Payment Method"],
        "Options chosen": [str(i[0]) for i in data.values()]
    }, index=[i for i in range(1, 8)])

    # update session states
    st.session_state.choices = to_show
    st.session_state.pred = model.predict_proba(final_df)[0][1]


# for spacing
st.write("")
st.write("Your chosen options and prediction will appear below:")

if 'choices' in st.session_state:
    st.dataframe(st.session_state.choices)

if 'pred' in st.session_state:
    if st.session_state.pred > 0.385:
        st.error("### :material/close: The customer might churn")
    else:
        st.success("### :material/task_alt: The customer will most likely stay")
