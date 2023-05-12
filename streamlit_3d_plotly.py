import streamlit as st
import numpy as np
import pandas as pd
import random
import plotly.graph_objects as go
import plotly.express as px

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPRegressor

def fit_a_metamodel(data,train_per,name,meta_params,in_axes,out_axes):
    st.write(f"Fitting {name} metamodel with given params and {train_per} % train data... sampled randomly...")
    st.write(f"input axes = {in_axes}, performance measure (out axes) = {out_axes}")
    num_samples_percent = train_per/100 # num of sample points is num_samples_percent % of whole data we created above
    
    # randomly sample some K training points
    k = int(data.shape[0]*num_samples_percent)
    indices = random.sample(range(0, data.shape[0]), k)
    X_train = data[in_axes].values[indices]
    y_train = data[out_axes].values[indices]
    
    if(name=="GPR"):
        # let us fit a GPR metamodel with RBF kerneli
        kernel = RBF(length_scale=meta_params[0],length_scale_bounds=meta_params[1])
        regr = GaussianProcessRegressor(kernel=kernel,random_state=0).fit(X_train,y_train)
    elif(name=="NN"):
        regr = MLPRegressor(hidden_layer_sizes=np.repeat(meta_params[1],meta_params[0]),
                            activation='relu',
                            solver='lbfgs',
                            random_state=1,
                            max_iter=3000).fit(X_train, y_train)
    st.write("fitted!")
    st.markdown("""---""")
    st.write("Select the components to visualize as a 3D slice (Metamodel):")

    # make predictions on all data
    gx = regr.predict(data[in_axes].values)
    col1, col2 = st.columns(2)
    with col1:
        selected_x_axis = st.selectbox("x-axis-meta", in_axes)
    with col2:
        selected_y_axis = st.selectbox("y-axis-meta", [a for a in in_axes if (a!=selected_x_axis)])
    selected_function = out_axes
    # making two columns 20% 80% of page
    col1, col2 = st.columns([20,80])
    # this 20% col is for sliders
    with col1:
        other_axes = [i for i in in_axes if i not in [selected_x_axis, selected_y_axis]]
        values = []
        if other_axes:
            st.write("Select the range of values for the other axes")
            for axis in other_axes:
                min_value = float(data[axis].min())
                max_value = max(float(data[axis].max()), min_value+1)
                total_values = np.unique(data[axis].values)
                step = (max_value - min_value)/(len(total_values)-1)
                val = st.slider(label = f"{axis} range_", min_value=min_value, max_value=max_value, step=step)
                values.append(val)
    # this 80% col is for 3D plot
    with col2:
        # find the slice
        data_slice = data
        gx_temp = gx
        for i, axis in enumerate(other_axes):
            gx_temp = gx_temp[data_slice[axis] == values[i]]
            data_slice = data_slice[data_slice[axis] == values[i]]
        if(len(data_slice)==0):
            st.warning(f"The selected slice contains {len(data_slice)} points")
        else:
            st.info(f"The selected slice contains {len(data_slice)} points")
        unique_vals = len(total_values)
        z=np.array(gx_temp)
        z=np.resize(z,(unique_vals,unique_vals))
        x=np.array((data_slice[selected_x_axis].values))
        x=np.resize(x,(unique_vals,unique_vals))
        y=np.array((data_slice[selected_y_axis].values))
        y=np.resize(y,(unique_vals,unique_vals))
        
        fig = go.Figure(data=[go.Surface(z=z,x=x,y=y)])
        fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
        fig.update_layout(title='Slice at given SD,sD', autosize=False,width=500, height=500)
        st.plotly_chart(fig, use_container_width=True)
    
def main():

    st.header("Visulaizing Supply chain data")
    # upload a file here
    uploaded_file = st.file_uploader("Upload a file")

    if uploaded_file is not None:
        # load the data in pandas dataframe
        df = pd.read_csv(uploaded_file)
        # get column names 
        column_names = list(df.columns)
        # defining defaults for multiselect below
        if("supplychain" in str(uploaded_file)):
            default_axes = column_names[5:13]
            default_functions = column_names[13:19]
        else:
            default_axes = column_names[0:2]
            default_functions = column_names[3:]
        # select input parameter columns
        axes = st.multiselect("Columns containing coordinates along each axis:", column_names, default=default_axes)
        # select output performance measure columns
        functions = st.multiselect("Columns containing function values:", default_functions, default=default_functions)

        # following code snippet is for visualizing the supply chain data (it is already recorded by running sims)
        
        st.write("Select the components to visualize as a 3D slice:")
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_x_axis = st.selectbox("x-axis", axes, index=0)
        with col2:
            selected_y_axis = st.selectbox("y-axis", [a for a in axes if (a!=selected_x_axis)], index=0)
        with col3:
            selected_function = st.selectbox("function component", functions, index=0)
        if selected_x_axis == selected_y_axis or selected_y_axis==selected_function or selected_x_axis == selected_function:
            st.error("Please select two distinct axes to visualize the meta-model")

        # make sliders for all unselected input parameter columns and plot the given slice
        st.markdown("""---""")
        # making two columns 20% 80% of page
        col1, col2 = st.columns([20,80])
        # this 20% col is for sliders
        with col1:
            other_axes = [i for i in axes if i not in [selected_x_axis, selected_y_axis]]
            values = []
            if other_axes:
                st.write("Select the range of values for the other axes")
                for axis in other_axes:
                    min_value = float(df[axis].min())
                    max_value = max(float(df[axis].max()), min_value+1)
                    total_values = np.unique(df[axis].values)
                    step = (max_value - min_value)/(len(total_values)-1)
                    val = st.slider(label = f"{axis} range", min_value=min_value, max_value=max_value, step=step)
                    values.append(val)
        # this 80% col is for 3D plot
        with col2:
            # find the slice
            data_slice = df
            for i, axis in enumerate(other_axes):
                data_slice = data_slice[data_slice[axis] == values[i]]
            
            if(len(data_slice)==0):
                st.warning(f"The selected slice contains {len(data_slice)} points")
            else:
                st.info(f"The selected slice contains {len(data_slice)} points")
            unique_vals = len(total_values)
            z=np.array((data_slice[selected_function].values))
            z=np.resize(z,(unique_vals,unique_vals))
            x=np.array((data_slice[selected_x_axis].values))
            x=np.resize(x,(unique_vals,unique_vals))
            y=np.array((data_slice[selected_y_axis].values))
            y=np.resize(y,(unique_vals,unique_vals))
            
            fig = go.Figure(data=[go.Surface(z=z,x=x,y=y)])
            fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
            fig.update_layout(title='Slice at given SD,sD', autosize=False,width=500, height=500)
            #fig.show()
            st.plotly_chart(fig, use_container_width=True)
        
        st.header("Fitting a Metamodel")
        # making two columns of page
        col1, col2 = st.columns([50,50])
        # apply metamodels and optimizers
        with col1:
            # choose an metamodel to fit
            metamodel_name = st.radio(label = "Choose a Metamodel", options = ["GPR","NN"], index=0)
            st.write("How much data should I use to train the model?")
            train_data_per = st.slider(label = "train percent", min_value=25, max_value=100, step=25)
        with col2:
            if(metamodel_name=="GPR"):
                st.write("Select parameter values for kernel (RBF)")
                length = st.number_input(label="length", min_value=0.1, max_value=None)
                length_bounds = st.slider(label="length_bounds", min_value=1.0, max_value=100.0, value=(1.0, 100.0))
                meta_params = [length,length_bounds]
            else:
                st.write("Select parameter values NN Metamodel")
                n_layers = st.number_input(label="Number of hidden layers?", min_value=1, max_value=8)
                n_neurons = st.number_input(label="Number of neurons in a hidden layer?", min_value=8, max_value=32, step=8)
                meta_params = [n_layers,n_neurons]
        
        if st.button(label="Build a Metamodel"):
            regressor = fit_a_metamodel(data = df,train_per = train_data_per,name = metamodel_name,
                            meta_params = meta_params,in_axes = axes, out_axes = selected_function)
                
        
if __name__ == "__main__":
    main()