pattern='test*.py'

while getopts ":ht:" opt; do
  case $opt in
    h) echo "usage: $0 [-t] test type, e.g. benchmark or notebook, or without for default test*.py"; exit ;;
    t) pattern=$OPTARG'_test*.py' ;;
  esac
done

echo 'running '$pattern
python -m unittest discover -p $pattern
