function [s,r,e,em,ConvStep,E,EM] = FlogAna(name, ConvThres)

fin = fopen(name,'r');

s = nan(10,1);
r = nan(10,1);
e = nan(10,1);
em = nan(10,1);

iread = 0;
while (~feof(fin))
   line = fgetl(fin);
   readNums = sscanf(line, "%g%g%g%g");
   if(numel(readNums)==4)
       iread = iread + 1;
       s(iread) = readNums(1);
       r(iread) = readNums(2);
       e(iread) = readNums(3);
       em(iread) = readNums(4);
   end
end
fclose(fin);

eLast = e(end);
eMean = mean(e(max(end-10,1):end));
eMMean = mean(em(max(end-10,1):end));

convCri = abs(eLast-eMean)/eMean;

fprintf("read %d lines, error mean var = %.2e\n", numel(s), convCri);

E = eMean;
EM = eMMean;

EThres = eMean * ConvThres;

ConvStep = -1;
for i = 1:numel(e)
    if(e(i) <= EThres)
       ConvStep = s(i);
       break;
    end
end

