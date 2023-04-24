
compressY = 1e-3;
skewX = 0;
quadP1 = [-0.5, -0.5*compressY]';
quadP2 = [0.5 + skewX, -0.5*compressY]';

coords = [quadP1, quadP2, -quadP1, - quadP2];

cla;
patch(coords(1,:), coords(2,:), coords(1,:)); axis equal;

%%
vol = Int_Quad(0, @(pphy, pparam, i) 1, coords);
pc = Int_Quad([0,0]', @(pphy, pparam, i) pphy, coords);

for i = 1:4
    faceCoords{i} = coords(:, mod((i:i+1)-1,4)+1); 
    faceNorms{i} =[0 1; -1 0] * (faceCoords{i}(:,2)-faceCoords{i}(:,1));
    faceNorms{i} = faceNorms{i} / norm(faceNorms{i}); 
    faceAreas{i} = Int_Line(0, @(pphy, pparam, iGauss) 1, faceCoords{i}); 
    facenDir{i} = 2 * mean(faceCoords{i},2); 
    faceGweight{i} = 1; 
    faceTangWeight{i} = faceAreas{i}/norm(facenDir{i});
    faceTangWeight{i} = max(min(1,faceAreas{i}/norm(facenDir{i})),0.1); 
    faceTangWeight{i} = 1; 
    faceTangMod{i} = min(1,faceAreas{i}/norm(facenDir{i}));
    faceTangMod{i} = 1;
end


LR = max(coords - pc,[],2);
LR = max(LR) * (LR/max(LR)).^0;


Moment = Int_Quad(zeros(10,10), @(pphy, pparam, i) fDBV(pphy-pc,LR), coords);
Moment = Moment(1,:)/vol;



dirWeights = [1 1 1/2 1/6];
tangWeight = 1;
normed = true;

options.xyIso = true;


A = zeros(10,10);
for i = 1:4
   A = Int_Line(A, @(pphy, pparam, iGauss) faceFunctional(fDBV(pphy-pc,LR,Moment),fDBV(pphy-pc,LR,Moment),...
        dirWeights, faceGweight{i}, tangWeight * faceTangWeight{i}, faceTangMod{i}, facenDir{i}, normed, options),...
        faceCoords{i});
  
end
A = A(2:end,2:end)
cond(A)









function DBV = fDBV(p, LR, Moment)
    DBV = nan(10,10);
    dxs = [0,1,0,2,1,0,3,2,1,0];
    dys = [0,0,1,0,1,2,0,1,2,3];
    dxys = [dxs;dys];
    
    for idiff = 1:10
        for iBase = 1:10
            DBV(idiff, iBase) = poly2d(p./LR,dxys(:,iBase), dxys(:,idiff))/...
                (LR(1)^dxys(1,idiff) * LR(2)^dxys(2,idiff));
        end
    end
    if nargin == 3
       DBV(1,:) = DBV(1,:) - Moment; 
    end
    
end

function v = poly2d(p, npower, ndiff)
    dFactorials = [...
        1 0 0 0
        1 1 0 0
        1 2 2 0
        1 3 6 6];
    v = dFactorials(npower(1)+1, ndiff(1)+1) * dFactorials(npower(2)+1, ndiff(2)+1);
    if v ~= 0
       v = v * p(1)^(npower(1) - ndiff(1)) * p(2)^(npower(2) - ndiff(2)) ;
    end
        

end

function v = faceFunctional(DBVI, DBVJ, dirWeights, geomWeight, tangWeight, tangMod, nDir, normed, options)
%     DBVI(10,:)
    if normed
        n1 = nDir(1);
        n2 = nDir(2);
        t1 = nDir(2) * tangWeight;
        t2 = -nDir(1) * tangWeight;
        v = zeros(size(DBVI,2), size(DBVJ,2));
        for i = 1:size(v,1)
            for j = 1:size(v,2)
                v(i,j) = v(i,j) + DBVI(1,i) * DBVI(1,j) * (dirWeights(1) * geomWeight)^2;
                
                csumI = DBVI(2,i) * n1 + DBVI(3,i) * n2;
                csumJ = DBVJ(2,j) * n1 + DBVJ(3,j) * n2;
                v(i,j) = v(i,j) + csumI * csumJ * (dirWeights(2) * geomWeight)^2;
                csumI = DBVI(2,i) * t1 + DBVI(3,i) * t2;
                csumJ = DBVJ(2,j) * t1 + DBVJ(3,j) * t2;
                v(i,j) = v(i,j) + csumI * csumJ * (dirWeights(2) * geomWeight)^2 * tangMod;
                
                csumI = ...
                    DBVI(4,i) * n1 * n1 + ...
                    DBVI(5,i) * n1 * n2 * 2+ ...
                    DBVI(6,i) * n2 * n2;
                csumJ = ...
                    DBVJ(4,j) * n1 * n1 + ...
                    DBVJ(5,j) * n1 * n2 * 2 + ...
                    DBVJ(6,j) * n2 * n2;
                v(i,j) = v(i,j) + csumI * csumJ * (dirWeights(3) * geomWeight)^2;
                
                csumI = ...
                    DBVI(4,i) * n1 * t1 + ...
                    DBVI(5,i) * (n1 * t2 + n2 * t1) + ...
                    DBVI(6,i) * n2 * t2;
                csumJ = ...
                    DBVJ(4,j) * n1 * t1 + ...
                    DBVJ(5,j) * (n1 * t2 + n2 * t1) + ...
                    DBVJ(6,j) * n2 * t2;
                v(i,j) = v(i,j) + csumI * csumJ * (dirWeights(3) * geomWeight)^2 * 2* tangMod;
                
                csumI = ...
                    DBVI(4,i) * t1 * t1 + ...
                    DBVI(5,i) * t1 * t2 * 2+ ...
                    DBVI(6,i) * t2 * t2;
                csumJ = ...
                    DBVJ(4,j) * t1 * t1 + ...
                    DBVJ(5,j) * t1 * t2 * 2 + ...
                    DBVJ(6,j) * t2 * t2;
                v(i,j) = v(i,j) + csumI * csumJ * (dirWeights(3) * geomWeight)^2* tangMod;
                
                csumI = ...
                    DBVI(7,i) * n1 * n1 * n1 + ...
                    DBVI(8,i) * n1 * n1 * n2 * 3+ ...
                    DBVI(9,i) * n1 * n2 * n2 * 3+...
                    DBVI(10,i) *n2 * n2 * n2;
                csumJ = ...
                    DBVI(7,j) * n1 * n1 * n1 + ...
                    DBVI(8,j) * n1 * n1 * n2 * 3+ ...
                    DBVI(9,j) * n1 * n2 * n2 * 3+...
                    DBVI(10,j) *n2 * n2 * n2;
                v(i,j) = v(i,j) + csumI * csumJ * (dirWeights(4) * geomWeight)^2;
                
                csumI = ...
                    DBVI(7,i) * n1 * n1 * t1 + ...
                    DBVI(8,i) * (n1 * n1 * t2 + n1 * t1 * n2 + t1 * n1 * n2)+ ...
                    DBVI(9,i) * (n1 * n2 * t2 + n1 * t2 * n2 + t1 * n2 * n2)+...
                    DBVI(10,i) *n2 * n2 * t2;
                csumJ = ...
                    DBVI(7,j) * n1 * n1 * t1 + ...
                    DBVI(8,j) * (n1 * n1 * t2 + n1 * t1 * n2 + t1 * n1 * n2)+ ...
                    DBVI(9,j) * (n1 * n2 * t2 + n1 * t2 * n2 + t1 * n2 * n2)+...
                    DBVI(10,j) *n2 * n2 * t2;
                v(i,j) = v(i,j) + csumI * csumJ * (dirWeights(4) * geomWeight)^2 * 3* tangMod;
                
                csumI = ...
                    DBVI(7,i) * n1 * t1 * t1 + ...
                    DBVI(8,i) * (n1 * t1 * t2 + t1 * n1 * t2 + t1 * t1 * n2)+ ...
                    DBVI(9,i) * (n1 * t2 * t2 + t1 * n2 * t2 + t1 * t2 * n2)+...
                    DBVI(10,i) *n2 * t2 * t2;
                csumJ = ...
                    DBVI(7,j) * n1 * t1 * t1 + ...
                    DBVI(8,j) * (n1 * t1 * t2 + t1 * n1 * t2 + t1 * t1 * n2)+ ...
                    DBVI(9,j) * (n1 * t2 * t2 + t1 * n2 * t2 + t1 * t2 * n2)+...
                    DBVI(10,j) *n2 * t2 * t2;
                v(i,j) = v(i,j) + csumI * csumJ * (dirWeights(4) * geomWeight)^2 * 3* tangMod;
                
                csumI = ...
                    DBVI(7,i) * t1 * t1 * t1 + ...
                    DBVI(8,i) * t1 * t1 * t2 * 3+ ...
                    DBVI(9,i) * t1 * t2 * t2 * 3+...
                    DBVI(10,i) *t2 * t2 * t2;
                csumJ = ...
                    DBVI(7,j) * t1 * t1 * t1 + ...
                    DBVI(8,j) * t1 * t1 * t2 * 3+ ...
                    DBVI(9,j) * t1 * t2 * t2 * 3+...
                    DBVI(10,j) *t2 * t2 * t2;
                v(i,j) = v(i,j) + csumI * csumJ * (dirWeights(4) * geomWeight)^2* tangMod;
                
                
                
            end
        end
    else
        dirWeights = dirWeights.*norm(nDir).^[0,1,2,3];
        if options.xyIso
            weights = geomWeight * [dirWeights(1),dirWeights(2),dirWeights(2),dirWeights(3),sqrt(2)*dirWeights(3),dirWeights(3),...
            dirWeights(4),sqrt(3)*dirWeights(4),sqrt(3)*dirWeights(4),dirWeights(4)];
        else
            weights = geomWeight * [dirWeights(1),dirWeights(2),dirWeights(2),dirWeights(3),dirWeights(3),dirWeights(3),...
                dirWeights(4),dirWeights(4),dirWeights(4),dirWeights(4)];
        end
        v = DBVI' * diag(weights.^2) * DBVJ;
        
    end

end




