#!/bin/tcsh 
#
# Parent process
# by protopop@unh.edu 08/01/2000

set exitcode = 66
set parf = 0 
onintr -

echo "Compiling ..."
g++ cMain.cc -g -o cMain

while($exitcode == 66)

  onintr -
  if (-e core) rm core 
  cMain $parf Dan
  set exitcode = $?
  if ($exitcode == 66)  then
    g++ cMain.cc -g -o cMain >& /tmp/recompile.log
    set erf = `wc -l /tmp/recompile.log | cut -d/ -f1`
    if ($erf > 0) then
      echo " - An error occurred at recompilation."
    else 
      echo " - Okay. Recompilation done."
    endif
    set parf = 1
  else 
    set exitcode = 0	
  endif

end

#echo Definitely exited.
exit 0
