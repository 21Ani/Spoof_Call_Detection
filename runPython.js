const {spawn} = require(`child_process`);
const pythonScript = `./main.py`;
const pythonProcess = spawn(`python`, [`-u`,pythonScript]);
pythonProcess.stdout.on(`data`,data=>{
    console.log(data.toString());
})
pythonProcess.stderr.on(`data`,data=>{
    console.log(`error`);
})
pythonProcess.on(`close`,code=>{
    console.log(`child process exited with code ${code}`);
})