`timescale 1ns / 1ps

module tb_clock_generator;
    reg clk, clk_en;
    wire o_clk;

    always 
        #5 clk = ~clk;

    initial begin
        clk = 0;
        clk_en = 0;

        #100
            clk_en = 1;
        #100
            clk_en = 0;
        #100

    $finish;
    end

    clock_gating_model DUT(
        .i_clk(clk),
        .i_clk_en(clk_en),
        .o_clk(o_clk)
    );
endmodule