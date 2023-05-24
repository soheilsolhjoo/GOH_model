function ss_plot(dataset,name)
if istable(dataset)
    plot(dataset.Lambda11,dataset.Sigma11MPa,'.')
    hold on
    plot(dataset.Lambda22,dataset.Sigma22MPa,'.')
else
    plot(dataset(:,1),dataset(:,2),'.')
    hold on
    plot(dataset(:,3),dataset(:,4),'.')
end

title(name);
xlabel('Stretch')
ylabel('Stress (MPa)')
end