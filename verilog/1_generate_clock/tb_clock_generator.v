`timescale 1ns/1ps

module tb_clock_generator;

    reg clk;

    always 
        #5 clk = ~clk;

    initial begin
        $display ("initialize value [%d]", $time);
            clk <= 0;

        $display ("start [%d]", $time);
            #100 
        @display ("[%d]", $time);
        
        $finish;
    end

endmodule