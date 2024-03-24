`timescale 1ns/1ps

module tb_fsm_test;
    reg clk, reset_n;
    reg i_run;
    wire o_done;

    always #5 clk = ~clk;

    initial begin
        clk <= 0;
        reset_n <= 1;
        i_run <= 0;

        #100 reset_n <= 0;
        #10 reset_n <= 1;
        #10 @(posedge clk);
            i_run <= 1;
            @(posedge clk);
            i_run <= 0;

        #100 
            $finish;
    end

//call DUT
fsm_test u_fsm_test(
    .clk (clk),
    .reset_n (reset_n),
    .i_run(i_run),
    .o_done(o_done)
);
endmodule