#!/bin/zsh
# Kill any Codex task running longer than 10 minutes (stuck). Its task is retried by the next sweep.
secs() { local t=$1 d=0; [[ $t == *-* ]] && { d=${t%%-*}; t=${t#*-}; }; local a=(${(s/:/)t}); local s=0; for x in $a; do s=$((s*60+10#$x)); done; echo $((d*86400+s)); }
while pgrep -f "zsh .*supervise_v2.sh" >/dev/null || pgrep -f "zsh .*run_v2.sh" >/dev/null; do
  ps -eo pid=,etime=,command= | grep "codex exec" | grep -v grep | while read p e rest; do
    [[ $(secs $e) -gt 600 ]] && { echo "$(date +%H:%M:%S) killed stuck codex $p"; kill $p; }
  done
  sleep 60
done
