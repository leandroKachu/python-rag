Create virtual environment inside the folder:
```bash
python3 -m venv .venv
```
Create Alias on bashrc to start/stop venv

* Enter Bashrc
```bash
vim ~/.bashrc "linux environment"
```
* Set this alias
```bash
start-venv() {
  if [ -d ".venv" ]; then
    source .venv/bin/activate
  else
    echo ".venv are not found in this dir"
  fi
}

stop-venv() {
	if [ -d ".venv" ]; then
	  deactivate
	else
	  echo ".venv are not found in this dir"
	fi
}
```

* lib dependency
```bash
pip install faiss-cpu numpy sentence-transformers
pip install -U langchain langchain-google-genai google-generativeai
pip install scikit-learn 
pip install pdfplumber
pip install -U -q "google-genai"
pip install python-multipart
pip insall dotenv


pip install -U \
  langchain-core \
  langchain-community \
  langchain-google-genai


