`timescale 1ns/1ps

module tb_fsm_counter_test;
    reg clk, reset_n;
    reg is_run;
    reg [6:0] i_num_cnt;
    wire o_done;
    wire o_idle;
    wire o_running;
 
    always #5 clk = ~clk;

    initial begin
        clk <= 0;
        reset_n <=1; 
        is_run <= 0;
        i_num_cnt <= 0;
        #100 reset_n <= 0;
        #10 reset_n <= 1;
        #10 @(posedge clk)
            wait(o_idle);
            i_num_cnt <= 7'd100;
            is_run <= 1;
            @(posedge clk)
            is_run <= 0;
        wait(o_done);
        #100
        $finish;

    end
    fsm_counter_test dut(
    .clk(clk),
    .reset_n(reset_n), 
    .is_run(is_run),
    .i_num_cnt(i_num_cnt),
    .o_idle(o_idle),
    .o_running(o_running),
    .o_done(o_done)
    );
endmodule
 