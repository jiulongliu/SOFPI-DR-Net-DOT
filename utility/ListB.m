function iDataS=ListB(iD)
% iD=[112:116];
iDataS={};
for i=1:2^(length(iD))-1
    j=dec2bin(i,4);
    ind=find(j=='1');
    iDataS={iDataS{1:end},iD(ind)};
end
% return iDataS