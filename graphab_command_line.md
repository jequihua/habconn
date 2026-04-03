(.venv) PS C:\Users\dev\work\tum\habconn> & "C:\Program Files\Java\jdk-17\bin\java.exe" -jar tools\graphab.jar --help

Usage :
java -jar graphab.jar --metrics
java -jar graphab.jar [-proc n] --create prjname landrasterfile habitat=code1,...,coden [nomerge] [nodata=val] [minarea=val] [maxsize=val] [con8] [dir=path]
java -jar graphab.jar [-mpi | -proc n] [-nosave] [-distconv excost=val] --project prjfile.xml command1 [command2 ...]

Commands list :
--show
--dem rasterfile
--linkset distance=euclid|cost [name=linkname] [complete] [maxcost=valcost] [slope=coef] [remcrosspath|nopathsaved] [[code1,..,coden=cost1 ...] codei,..,codej=min:inc:max | extcost=rasterfile]
--uselinkset linkset1,...,linksetn
--removelinkset [linkset1,...,linksetn]
--corridor maxcost=[{]min:inc:max[}] [format=raster|vector] [beta=exp|var=name d=val p=val [min=val]]
--graph [name=graphname] [nointra] [threshold=[{]min:inc:max[}]]
--usegraph graph1,...,graphn
--removegraph [graph1,...,graphn]
--cluster d=val p=val [beta=val] [nb=val]
--pointset pointset.shp id=fieldname [name=pointname] [random_absence=value [inpatch|outpatch[=dist]]]
--usepointset pointset1,...,pointsetn
--removepointset [pointset1,...,pointsetn]
--pointdistance type=space|graph distance=leastcost|circuit|flow|circuitflow [dist=val proba=val]
--capa [area [exp=value] [code1,..,coden=weight ...]] | [file=capacity.csv id=fieldname capa=fieldname] | [maxcost=[{]valcost[}] codes=code1,code2,...,coden [weight]]
--gmetric global_metric_name [resfile=file.txt] [maxcost=valcost] [param1=[{]min:inc:max[}] [param2=[{]min:inc:max[}] ...]]
--cmetric comp_metric_name [maxcost=valcost] [param1=[{]min:inc:max[}] [param2=[{]min:inc:max[}] ...]]
--lmetric local_metric_name [maxcost=valcost] [param1=[{]min:inc:max[}] [param2=[{]min:inc:max[}] ...]]
--interp name resolution var=patch_var_name d=val p=val [multi=dist_max [sum]]
--model variable distW=[{]min:inc:max[}] [vars=var1,...,varn] [raster=r1,...,rn]
--delta global_metric_name [maxcost=valcost] [param1=[{]val[}] ...] obj=patch|link [sel=id1,id2,...,idn|fsel=file.txt]
--addpatch npatch global_metric_name [maxcost=valcost] [param1=val ...] gridres=min:inc:max [capa=capa_file] [multi=nbpatch,size] | patchfile=file.shp [capa=capa_field]
--remelem nstep global_metric_name [maxcost=valcost] [param1=val ...] obj=patch|link [sel=id1,id2,...,idn|fsel=file.txt]
--gtest nstep global_metric_name [maxcost=valcost] [param1=val ...] obj=patch|link sel=id1,id2,...,idn|fsel=file.txt
--gremove global_metric_name [maxcost=valcost] [param1=val ...] [patch=id1,id2,...,idn|fpatch=file.txt] [link=id1,id2,...,idm|flink=file.txt]
--metapatch [mincapa=value]
--landmod zone=filezones.shp id=fieldname code=fieldname [sel=id1,id2,...,idn ] [novoronoi]

min:inc:max -> val1,val2,val3...