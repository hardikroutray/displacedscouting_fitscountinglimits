universe = vanilla
initialdir = .
use_x509userproxy = true
error = ./con_logs_1/run_$(Process).error
log = ./con_logs_1/run_$(Process).log
output = ./con_logs_1/run_$(Process).out
executable = submit_limits.sh
arguments = $(Process)
Notification=never
queue 2