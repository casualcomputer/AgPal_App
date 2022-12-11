# AgPal_App

**This colab notebook shows the stuff I tried so far**: http://bit.ly/3FgL8qJ

This Streamlit app for AgPal:
* Shows the results of our semantic search 
* Shows the results of the AgPal searcch
* Allows users to identify and report bad matches by indices
  * Based on user input, prints out performance comparison
  * Stores user feedback into an Excel sheet (search log): https://bit.ly/3W3Yyh7
  
Features in-process:
* Refine text preprocessing 
* Modifying the table for user feedback to upload performance comparison data
* Adding more models to the interface
* Fine tuning models for semantic search, lexical search; define ranking function

Note, for the Google Drive feature to work, you need to create have a .streamlit/secrets.toml within your project directory.

