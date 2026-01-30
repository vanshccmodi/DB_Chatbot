
import streamlit as st
import pandas as pd

def render_visualization(results, key_prefix):
    """Render data tables and visualizations from SQL results."""
    if not results:
        return

    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    with st.expander("📊 Results & Visualization", expanded=False):
        tab_data, tab_summary, tab_viz = st.tabs(["📄 Data", "🧮 Summary", "📈 Visualize"])
        
        with tab_data:
            col_search, col_info = st.columns([3, 1])
            with col_search:
                search_term = st.text_input(
                    "🔍 Filter Data", 
                    placeholder="Type to search...",
                    label_visibility="collapsed",
                    key=f"{key_prefix}_search"
                )
            
            if search_term:
                # Filter dataframe (case-insensitive) across all columns
                mask = df.astype(str).apply(
                    lambda x: x.str.contains(search_term, case=False, na=False)
                ).any(axis=1)
                filtered_df = df[mask]
                with col_info:
                    st.caption(f"Showing {len(filtered_df)} / {len(df)} rows")
                st.dataframe(filtered_df, width="stretch")
            else:
                with col_info:
                    st.caption(f"Total rows: {len(df)}")
                st.dataframe(df, width="stretch")
            
        with tab_summary:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if not numeric_cols:
                st.info("No numeric columns found for summary statistics.")
            else:
                st.markdown("### 📊 Quick Statistics")
                
                # Create metrics for each numeric column
                for col in numeric_cols:
                    st.markdown(f"**{col}**")
                    c1, c2, c3, c4 = st.columns(4)
                    
                    series = df[col]
                    with c1:
                        st.metric("Total", f"{series.sum():,.2f}")
                    with c2:
                        st.metric("Average", f"{series.mean():,.2f}")
                    with c3:
                        st.metric("Min", f"{series.min():,.2f}")
                    with c4:
                        st.metric("Max", f"{series.max():,.2f}")
                    st.divider()

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
                    # Try to find a good categorical column for X axis
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
