function demo_gif()
filename = './clock.gif';
x = 0:0.1:100;
for i = 1:100
    plot(x,i*x/1000);
    ylim([0 10])
    save_gif(filename,i)
end

end
function save_gif(filename,n)
frame = getframe;
im = frame2im(frame);
[imind,cm] = rgb2ind(im,256);
% Write to the GIF File
if n == 1
    imwrite(imind,cm,filename,'gif', 'Loopcount',inf);
else
    imwrite(imind,cm,filename,'gif','WriteMode','append');
end

end