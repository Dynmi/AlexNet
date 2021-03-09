var files = system.path.scan(context.cmd_args[1])
var ofs = iostream.fstream("./images.list", iostream.openmode.out)
var id = 0
foreach it in files
    if it.type != system.path.type.dir
        ofs.println(to_string(id) + " " + context.cmd_args[1] + it.name)
    end
end