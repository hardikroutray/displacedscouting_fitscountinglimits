universe = vanilla
initialdir = .
use_x509userproxy = true
error = ./con_logs/run_$(Process).error
log = ./con_logs/run_$(Process).log
output = ./con_logs/run_$(Process).out
executable = submit_fitter.sh
arguments = $(Process)
Notification=never
queue 2