Create virtual environment inside the folder:
```bash
python3 -m venv .venv
```
Create Alias on bashrc to start/stop venv

* Enter Bashrc
```bash
vim ~/.bashrc
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

