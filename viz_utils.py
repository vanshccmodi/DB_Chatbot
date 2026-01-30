
import streamlit as st
import pandas as pd

def render_visualization(results, key_prefix):
    """Render data tables and visualizations from SQL results."""
    if not results:
        return

    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    with st.expander("📊 Results & Visualization", expanded=False):
        tab_data, tab_viz = st.tabs(["📄 Data", "📈 Visualize"])
        
        with tab_data:
            st.dataframe(df, use_container_width=True)
            
        with tab_viz:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
            
            if not numeric_cols:
                st.info("No numeric data found to visualize.")
            else:
                col1, col2, col3 = st.columns(3)
                with col1:
                    chart_type = st.selectbox(
                        "Chart Type", 
                        ["Bar", "Line", "Area", "Scatter"],
                        key=f"{key_prefix}_chart_type"
                    )
                with col2:
                    # Default X axis logic
                    x_options = df.columns.tolist()
                    default_x = categorical_cols[0] if categorical_cols else x_options[0]
                    # Find index safely
                    try:
                        def_index = x_options.index(default_x)
                    except ValueError:
                        def_index = 0
                        
                    x_axis = st.selectbox(
                        "X Axis", 
                        x_options, 
                        index=def_index,
                        key=f"{key_prefix}_x_axis"
                    )
                with col3:
                    y_axis = st.selectbox(
                        "Y Axis", 
                        numeric_cols, 
                        index=0,
                        key=f"{key_prefix}_y_axis"
                    )
                
                if chart_type == "Bar":
                    st.bar_chart(df, x=x_axis, y=y_axis)
                elif chart_type == "Line":
                    st.line_chart(df, x=x_axis, y=y_axis)
                elif chart_type == "Area":
                    st.area_chart(df, x=x_axis, y=y_axis)
                elif chart_type == "Scatter":
                    st.scatter_chart(df, x=x_axis, y=y_axis)
