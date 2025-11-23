import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import pandas as pd 
import streamlit as st
import sys
import os

pd.set_option('display.max_columns', 4)

def view_collections(dir):
    st.markdown("### DB Path: %s" % dir)

    try:
        client = chromadb.PersistentClient(path=dir)
        
        collections = client.list_collections()
        st.header("Collections")
        
        if not collections:
            st.warning("No collections found in this database.")
            return

        for collection in collections:
            try:
                data = collection.get()

                ids = data['ids']
                embeddings = data.get("embeddings", [])
                metadata = data.get("metadatas", [])
                documents = data.get("documents", [])

                # Create a more readable dataframe
                df_data = {
                    'id': ids,
                    'document': documents,
                }
                
                # Add metadata columns if available
                if metadata:
                    for i, meta in enumerate(metadata):
                        if isinstance(meta, dict):
                            for key, value in meta.items():
                                if key not in df_data:
                                    df_data[key] = [None] * len(ids)
                                df_data[key][i] = value
                
                df = pd.DataFrame(df_data)
                
                st.markdown(f"### Collection: **{collection.name}** ({len(ids)} items)")
                
                # Show summary stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Items", len(ids))
                with col2:
                    st.metric("Has Embeddings", "Yes" if embeddings else "No")
                with col3:
                    st.metric("Has Metadata", "Yes" if metadata else "No")
                
                # Display the dataframe
                st.dataframe(df, use_container_width=True, height=400)
                
                # Search functionality
                st.markdown("#### Search")
                search_term = st.text_input(f"Search in {collection.name}", key=f"search_{collection.name}")
                if search_term:
                    filtered_df = df[df['document'].str.contains(search_term, case=False, na=False) | 
                                    df['id'].str.contains(search_term, case=False, na=False)]
                    st.dataframe(filtered_df, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error loading collection {collection.name}: {str(e)}")
                
    except Exception as e:
        st.error(f"Error connecting to database: {str(e)}")
        st.info("Make sure the database path is correct and the database exists.")

# Main Streamlit app
st.title("ChromaDB Viewer")

# Try to get path from command line arguments first
db_path = None
if len(sys.argv) > 1:
    db_path = sys.argv[1]

# If not provided, use text input or default
if not db_path:
    # Default path based on your project structure
    default_path = r"E:\aie2\aie-hackathon-chromadb-scripts\chroma_db"
    db_path = st.text_input("Database Path", value=default_path)

if db_path:
    if os.path.exists(db_path):
        view_collections(db_path)
    else:
        st.error(f"Database path does not exist: {db_path}")
        st.info("Please enter a valid path to your ChromaDB database directory.")
else:
    st.info("Please enter a database path to view collections.")