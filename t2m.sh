while true
do
 python train-text2mel.py || echo "App crashed... restarting..." >&2
 echo "Press Ctrl-C to quit." && sleep 1
done