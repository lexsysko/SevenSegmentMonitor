#!/bin/sh

# Check for 'free-threading' in sys.version or hasattr sys._is_gil_enabled
if python -c "import sys; exit(0 if 'free-threading' in sys.version else 1)"; then
  echo "Free-threaded build detected. Disabling GIL..."
  export PYTHON_GIL=0
fi



if [ "${WAITER:-}" = "1" ]; then
 echo "RUNNING TERMINAL INFINITE WAITER..."
 exec tail -f /dev/null
 exit
fi

#if [ -n "${LOGLEVEL:-WARNING}" ]; then
#  LOGLEVEL=" --loglevel ${LOGLEVEL}"
#fi

if [ -n "${TUNE_VIDEO_BRIGHTNESS:-}" ]; then
  echo "Setup webcam video brightness to ${TUNE_VIDEO_BRIGHTNESS}"
  v4l2-ctl -d /dev/video0 --set-ctrl=brightness=${TUNE_VIDEO_BRIGHTNESS}
fi


echo "\n\nRUNNING  ${LOGLEVEL}${THRESHOLD:-}..."
exec python /app/src/SevenSegmentMonitor/main.py