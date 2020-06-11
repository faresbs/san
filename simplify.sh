#!/bin/bash

if [ -z "$1" ];then

echo "simplify.sh <path>"
exit 0
fi

path=$1

## declare an array variable
declare -a arr=("valid" "test")

## now loop through the above array
for i in "${arr[@]}"
do
	echo $i
	ref_file="${path}/references_${i}.txt"
	hyp_file="${path}/translations_${i}.txt"

	simpl_ref_file="${path}/simpl_references_${i}.txt"
	simpl_hyp_file="${path}/simpl_translations_${i}.txt"

	echo $ref_file
	echo $hyp_file
	echo $simpl_hyp_file
	echo $simpl_ref_file

	# apply some simplifications to the recognition
	cat ${hyp_file} | sed -e 's,loc-,,g' -e 's,cl-,,g' -e 's,qu-,,g' -e 's,poss-,,g' -e 's,lh-,,g' -e 's,S0NNE,SONNE,g' -e 's,HABEN2,HABEN,g'|sed -e 's,__EMOTION__,,g' -e 's,__PU__,,g'  -e 's,__LEFTHAND__,,g' |sed -e 's,WIE AUSSEHEN,WIE-AUSSEHEN,g' -e 's,ZEIGEN ,ZEIGEN-BILDSCHIRM ,g' -e 's,ZEIGEN$,ZEIGEN-BILDSCHIRM,' -e 's,^\([A-Z]\) \([A-Z][+ ]\),\1+\2,g' -e 's,[ +]\([A-Z]\) \([A-Z]\) , \1+\2 ,g'| sed -e 's,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g'|sed -e 's,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g'|  sed -e 's,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g'|sed -e 's,\([ +]SCH\) \([A-Z][ +]\),\1+\2,g'|sed -e 's,\([ +]NN\) \([A-Z][ +]\),\1+\2,g'| sed -e 's,\([ +][A-Z]\) \(NN[ +]\),\1+\2,g'| sed -e 's,\([ +][A-Z]\) \([A-Z]\)$,\1+\2,g'|  sed -e 's,\([A-Z][A-Z]\)RAUM,\1,g'| sed -e 's,-PLUSPLUS,,g' | perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| grep -v "__LEFTHAND__" | grep -v "__EPENTHESIS__" | grep -v "__EMOTION__" > ${simpl_hyp_file} 

	# apply some simplifications to the ground truth references
	cat ${ref_file} | sed -e 's,loc-,,g' -e 's,cl-,,g' -e 's,qu-,,g' -e 's,poss-,,g' -e 's,lh-,,g' -e 's,S0NNE,SONNE,g' -e 's,HABEN2,HABEN,g'|sed -e 's,__EMOTION__,,g' -e 's,__PU__,,g'  -e 's,__LEFTHAND__,,g' |sed -e 's,WIE AUSSEHEN,WIE-AUSSEHEN,g' -e 's,ZEIGEN ,ZEIGEN-BILDSCHIRM ,g' -e 's,ZEIGEN$,ZEIGEN-BILDSCHIRM,' -e 's,^\([A-Z]\) \([A-Z][+ ]\),\1+\2,g' -e 's,[ +]\([A-Z]\) \([A-Z]\) , \1+\2 ,g'| sed -e 's,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g'|sed -e 's,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g'|  sed -e 's,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g'|sed -e 's,\([ +]SCH\) \([A-Z][ +]\),\1+\2,g'|sed -e 's,\([ +]NN\) \([A-Z][ +]\),\1+\2,g'| sed -e 's,\([ +][A-Z]\) \(NN[ +]\),\1+\2,g'| sed -e 's,\([ +][A-Z]\) \([A-Z]\)$,\1+\2,g'|  sed -e 's,\([A-Z][A-Z]\)RAUM,\1,g'| sed -e 's,-PLUSPLUS,,g' | perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| grep -v "__LEFTHAND__" | grep -v "__EPENTHESIS__" | grep -v "__EMOTION__" > ${simpl_ref_file}

done