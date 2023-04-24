coords = [...
    -1 0 0 -1
    0 0 1 1];

cla;
patch(coords(1,:), coords(2,:), coords(1,:));


%%
vol = Int_Quad(0, @(pphy, pparam, i) pphy(1), coords)
xc = Int_Quad(0, @(pphy, pparam, i) pphy(1), coords)
yc = Int_Quad(0, @(pphy, pparam, i) pphy(2), coords)

area1 = Int_Line(0, @(pphy, pparam, i) 1, coords(:,1:2))