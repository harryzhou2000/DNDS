function r = Int_Line(r, f, coords)

pGauss = [-0.7745966692414833, 0, 0.7745966692414833];
wGauss = [5. / 9., 8. / 9., 5. / 9.];

J = norm(coords(:,1) - coords(:,2))/2;

for i = 1:numel(pGauss)
    pPhy =  (coords(:,2) - coords(:,1)) * (pGauss(i) + 1)/2 + coords(:,1);
    
    
    r = r + f(pPhy, pGauss(i), i) * wGauss(i) * J;
end



end