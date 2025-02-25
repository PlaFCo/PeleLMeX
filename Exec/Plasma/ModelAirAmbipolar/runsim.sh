#!/bin/bash
#
# Run ambipolar test case with all models
#
./PeleLMeX2d.gnu.MPI.EF.ex input.2d-regt_efambipolar &
./PeleLMeX2d.gnu.MPI.EF.ex input.2d-regt_efglobal &
./PeleLMeX2d.gnu.MPI.EF.ex input.2d-regt_eflocal &
./PeleLMeX2d.gnu.MPI.EF.ex input.2d-regt_efoskam &
./PeleLMeX2d.gnu.MPI.EF.ex input.2d-regt_efneutral &
./PeleLMeX2d.gnu.MPI.ex input.2d-regt_simplediff &
wait
echo "--------"
echo "Finished"

