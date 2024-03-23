`timescale 1ns/1ps
module tb_power_of_8;

    reg clk, reset_n;
    reg [7:0] i_a; 
    wire [31:0] o_output;

    integer  i;

    always #5 clk = ~clk;
    initial begin
        clk <= 0;
        reset_n <=1;
        i_a <= 0;
        #20
            reset_n <= 0;
        #20
            reset_n <= 1;
        #30 
            @(posedge clk);
                for (i = 0; i < 10; i = i+1) begin
                    @(negedge clk);
                    i_a <= i;
                    @(posedge clk);
                end
                @(negedge clk);
                i_a <=0;

        #200
            $finish;

    end
    power_of_8 dut(
        .clk(clk), 
        .reset_n(reset_n),
        .i_a(i_a),
        .o_output(o_output)
    );

endmodule
