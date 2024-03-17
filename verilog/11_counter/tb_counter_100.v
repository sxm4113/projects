`timescale 1ns/1ps
module tb_counter_100;

    reg clk, reset_n; 
    wire [7:0] o_value;
    wire [7:0] o_value_always;

    always #5 clk = ~clk;
    initial begin
        clk <= 0;
        reset_n <= 1;
        #20
            reset_n <= 0;
        #20
            reset_n <= 1;
        #2000
        $finish;
    end

    counter_100 dut(
    .clk(clk), 
    .reset_n(reset_n),
    .o_counter(o_value),
    .o_counter_always(o_value_always)
);
endmodule