function save_current_figure(filename, resolution, filetype)
set(gcf, 'InvertHardCopy', 'off');
set(gcf, 'Color', [1,1,1]);
if (strcmp(resolution, 'high'))
    export_fig(filename, filetype, '-rgb', '-r300', '-painters');
elseif (strcmp(resolution, 'low'))
    export_fig(filename, filetype, '-rgb', '-r100', '-painters');
else
    error('Error: resolution should be "high" or "low"')
end
end