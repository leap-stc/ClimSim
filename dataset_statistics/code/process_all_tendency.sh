i#!/bin/bash
directory="/ocean/projects/atm200007p/sungduk/hugging/E3SM-MMF_ne4/train/"
yearlist=(0001 0002 0003 0004 0005 0006 0007 0008 0009)
monlist=(01 02 03 04 05 06 07 08 09 10 11 12)
varlist=(state_q0001 state_q0002 state_q0003 state_t)

# export NETCDF_ROOT=/ocean/projects/atm200007p/shared/netcdf
# export PATH=/jet/home/walrus/nco/bin:$PATH
for varin in "${varlist[@]}"
do
for ilev in {00..59}; do	
sed -e "s/vvvvvvv/${varin}/g" -e "s/lllllll/${ilev}/g" tendency_vvvv_llll.py > tend_${varin}_${ilev}.py
chmod 700 tend_${varin}_${ilev}.py
rm submit_tend_${varin}_${ilev}.sh
cat <<EOF >> submit_tend_${varin}_${ilev}.sh
#!/bin/bash
#SBATCH --job-name="tend_${varin}_${ilev}_lev"
#SBATCH --output="logs/tend_${varin}_${ilev}_lev.%j.out"
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --export=ALL
#SBATCH --account=m3312
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liranp@uci.edu
#SBATCH -t 05:00:00

module load python
./tend_${varin}_${ilev}.py

EOF
sbatch submit_tend_${varin}_${ilev}.sh
done
done

