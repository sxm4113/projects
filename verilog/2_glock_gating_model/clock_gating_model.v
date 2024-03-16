`timescale 1ns/1ps

module clock_gating_model (
    input i_clk,
    input i_clk_en,
    output o_clk
);
    assign o_clk = i_clk & i_clk_en;
    
endmodule