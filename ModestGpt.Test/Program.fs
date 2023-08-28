namespace ModestGpt

open System
open System.IO

module Program =

    let text = File.ReadAllText("Test.txt")
    let dtStart = DateTime.Now
    let encoder = Encoder.create 1024 text
    printfn $"{DateTime.Now - dtStart}"
